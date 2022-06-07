from ast import arg
from audioop import avg
from cgi import test
from tkinter import image_names
import numpy as np
import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import cv2
from tqdm import tqdm
from utils.load_llff import load_llff_data
from scipy.spatial.transform import Rotation as R

from pprint import pprint

def get_subdirs(path) -> list:
        """ Check if the provided path contains the images or has other inner directories that 
            contain the images.

            e.g.: dataset           Has the following subdirs: ["train", "test", "val"]
                    |-> train
                    |-> test
                    |-> val

                  dataset           Does not have subdirs.
                    |-> img0.png
                    |-> img1.png
                    ...

                  dataset           Does not have subdirs and will use the images in dataset/
                    |-> docs
                    |-> img0.png
                    ...

        Args:
            path (str): Data path

        Returns:
            list: list of subdirs, if there are no subdirs will return a list with ['.']
        """
        subdirs = []

        if path[-1] != '/':
            path = path+'/'

        if not any(fname.endswith('.png') for fname in os.listdir(path)):
            subdirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

        if len(subdirs) == 0:
            subdirs = ['.'] 
        
        return subdirs

def batchify(locations, view_dirs, chunksize):
    locs_flat = torch.reshape(locations, [-1, locations.shape[-1]])
    view_dirs = torch.broadcast_to(view_dirs, locations.shape)
    view_dirs_flat = torch.reshape(view_dirs, [-1, view_dirs.shape[-1]])
    for i in range(0, locs_flat.shape[0], chunksize):
        #upper = i+chunksize if i+chunksize < locs_flat.shape[0] else locs_flat.shape[0]
        #print(f"{i}:{i+chunksize}, {locs_flat[i:i+chunksize].shape}, {locs_flat.shape[0]}")
        yield (locs_flat[i:i+chunksize], view_dirs_flat[i:i+chunksize])

class NeRFSubDataset():
    def __init__(self, rays, images, hwf, name, batch_size=None, shuffle=False):
        self.dataset = TensorDataset(rays, images)
        self.name = name
        self.hwf = hwf
        height, width, focal = hwf
        self.n_samples = images.shape[0]

        if batch_size is None:
            batch_size=height*width

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.iterator = iter(self.dataloader)

    def next_batch(self):
        try:
            batch_rays, target_rgb = next(self.iterator)
            
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch_rays, target_rgb = next(self.iterator)

        return batch_rays, target_rgb
        
class NeRFDataset():

    def __init__(self, 
                 args,
                 device=torch.device('cuda')):

        self.device = device
        self.subdirs_indices = [[0]]
        self.subdirs = []

        if args.dataset_type == "tiny":
            self.load_tiny(args.dataset_path, device=device)
        elif args.dataset_type == "synthetic":
            self.load_data(args.dataset_path, device=device)
        elif args.dataset_type == "llff":
            self.load_llff(args.dataset_path, device=device, factor=args.factor, spherify=args.spherify, test_path=args.test_path)
        elif args.dataset_type == "meshroom":
            self.load_meshroom(args.dataset_path, device=device)
        elif args.dataset_type == "colmap":
            self.load_colmap(args.dataset_path, device=device)
        
        self._create()
    
    def _create(self):
        #focal_length = focal_length.to(self.poses)
        
        rays_od = [torch.cat(self.get_rays_origins_and_directions(c2w, self.hwf), dim=-1) for c2w in tqdm(self.poses, unit="pose", desc="Generating rays")]
        rays_od = torch.stack(rays_od)
        
        print("Computing view dirs...")
        view_dirs = rays_od[..., 3:]
        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
        rays_odv = torch.cat([rays_od, view_dirs], dim=-1)
        
        self.subdatasets = []

        for indices, name in zip(self.subdirs_indices, self.subdirs):
            rays = torch.reshape(rays_odv[indices], [rays_odv[indices].shape[0], -1, 9])#.to(self.device)
            images = torch.reshape(self.images[indices], [self.images[indices].shape[0], -1, 3])#.to(self.device)
            self.subdatasets.append(NeRFSubDataset(rays, images, self.hwf, name))

        #focal_length = focal_length.to(self.device)

    def get_rays_od(self, img, subdataset=0):
        return self.subdatasets[subdataset].dataset[img][0][..., :6]

    def get_n_samples(self, subdataset=0):
        return self.subdatasets[subdataset].n_samples

    @staticmethod
    def get_rays_origins_and_directions(c2w, hwf=None):
        height, width, focal_length = hwf

        R = c2w[:3, :3]
        t = c2w[:3, 3:4].squeeze()

        # Obtain each pixel coordinates (u, v)
        uu, vv = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        uu, vv = uu.to(focal_length).transpose(-1, -2), vv.to(focal_length).transpose(-1, -2)

        # Pass from pixel coordinates to image plane (uu - width * .5) and (vv - height * .5)
        # and from image plane to 3D coordinates in the camera frame (/focal_length)
        xx = (uu - width * .5) / focal_length
        yy = (vv - height * .5) / focal_length
        
        # R matrix is in the form [down right back] instead of [right up back] 
        # which is why we must fix it.
        #
        # Given the assumptions above, we are going to create, for each pixel
        # its corresponding direction vector considering the camera's point of 
        # view (i.e., the R matrix)

        directions = torch.stack([xx, -yy, -torch.ones_like(xx)], dim=-1)
        #directions = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        
        rays_d = torch.sum(directions[..., None, :] * R, dim=-1)
        rays_o = t.expand(rays_d.shape)
        
        return rays_o, rays_d

    def next_batch(self, dataset="train"):
        subdataset = [d for d in self.subdatasets if d.name == dataset]
        return subdataset.next_batch()
    
    def load_data(self,  
                  dataset_path,
                  device=torch.device('cuda'), 
                  save_to=None):
        if dataset_path[-1] == '/':
            dataset_path = dataset_path[:-1]
        
        poses = None
        images = None
        subdirs = get_subdirs(dataset_path)
        
        for i_dir, dir in enumerate(subdirs):
            with open(f"{dataset_path}/transforms_{dir}.json") as json_file:
                data = json.load(json_file)

                camera_angle_x = data["camera_angle_x"]
                i = 0
                for frame in tqdm(data["frames"], unit="frame", leave=False, desc="Loading frames" if len(subdirs)==1 else f"Loading {dir} frames"):
                    # Load poses
                    pose = np.asarray(frame["transform_matrix"])
                    if poses is None:
                        poses = pose[None, ...]
                    else:
                        poses = np.concatenate((poses, pose[None, ...]))

                    file = frame["file_path"].split('/')[-1]
                    
                    # Load images
                    img = cv2.imread(f'{dataset_path}/{dir}/{file}')[..., [2, 1, 0]] #bgr to rgb
                
                    if images is None:
                        images = img[None, ...]
                    else:
                        images = np.concatenate((images, img[None, ...]))

                    i += 1
                    if i >= 5:
                        break

            self.subdirs_indices[i_dir] = list(range(self.subdirs_indices[i_dir][0], images.shape[0]))
            self.subdirs_indices.append([images.shape[0]])
            self.subdirs.append(dir)

        self.subdirs_indices = self.subdirs_indices[:-1]

        num_images, height, width, num_channels = images.shape
        #focal_length = 138.8889
        focal_length = width/(2 * np.tan(camera_angle_x/2))

        if save_to is not None:
            with open(save_to, 'wb') as f:
                np.savez(f, images=images, poses=poses, focal=focal_length)
        
        self.images = torch.from_numpy(images).to(device)
        #poses[:,2,:] *= -1
        self.poses = torch.from_numpy(poses).to(device)
        focal_length = torch.from_numpy(np.array([focal_length])).to(device)
        print("Focal", focal_length)
        self.hwf = (height, width, focal_length)

    def load_tiny(self, dataset_path, device=torch.device('cuda')):
        self.subdirs_indices = [list(range(0, 100)), list(range(100, 102)), list(range(102, 106))]
        self.subdirs = ["train", "val", "test"]

        data = np.load(dataset_path)

        # Images - shape: (num_images, height, width, channels)
        self.images = torch.from_numpy(data["images"]).to(device)
        num_images, height, width, num_channels = self.images.shape

        # Camera extrinsics
        self.poses = torch.from_numpy(data["poses"]).to(device)

        # Focal length (intrinsics)
        focal_length = torch.from_numpy(data["focal"]).to(device)
        print("Focal", focal_length)

        self.hwf = (height, width, focal_length)
       
    def load_llff(self, dataset_path, device=torch.device('cuda'), factor=1, spherify=False, test_path=None):
        poses = None
        images = None
        subdirs = get_subdirs(dataset_path)
        
        for i_dir, dir in enumerate(subdirs):
            images_dir, poses_dir, bds = load_llff_data(dataset_path, factor, bd_factor=.75, filename=f"poses_bounds_{dir}.npy")
            hwf = poses_dir[0, :3, -1]
            poses = poses_dir[:, :3, :4]

            if poses is None:
                poses = poses_dir[None, ...]
                images = images_dir[None, ...]
            else:
                poses = np.concatenate((poses, poses_dir[None, ...]))
                images = np.concatenate((images, images_dir[None, ...]))

            self.subdirs_indices[i_dir] = list(range(self.subdirs_indices[i_dir][0], images.shape[0]))
            self.subdirs_indices.append([images.shape[0]])
            self.subdirs.append(dir)

        self.images = torch.from_numpy(images).to(device)
        self.poses = torch.from_numpy(poses).to(device)
        focal_length = torch.from_numpy(np.array([hwf[2]])).to(device)
        self.hwf = (int(hwf[0]), int(hwf[1]), focal_length)

    def get_test_poses(self, i):
        return self.test_data[0][i], self.test_data[1][i]

    def load_colmap(self, dataset_path, device=torch.device('cuda')):
        if dataset_path[-1] == '/':
            dataset_path = dataset_path[:-1]

        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        poses = None
        images = None
        subdirs = get_subdirs(dataset_path)
        
        for i_dir, dir in enumerate(subdirs):
            with open(dataset_path+f'/images_{dir}.txt', 'r') as f:
                for i, line in enumerate(f):
                    if line[0]!='#' and i%2 == 0:
                        parts = line.split(" ")

                        # Load poses
                        qvec = np.array(tuple(map(float, parts[1:5])))
                        tvec = np.array(tuple(map(float, parts[5:8])))

                        r = R.from_quat(qvec)
                        rot = r.as_matrix()
                        t = tvec.reshape([3, 1])
                        m = np.concatenate([np.concatenate([rot, t], 1), bottom], 0)
                        print(m)
                        c2w = np.linalg.inv(m)
                        c2w[0:3, 2] *= -1  
                        c2w[0:3, 1] *= -1

                        if poses is None:
                            poses = c2w[None, ...]
                        else:
                            poses = np.concatenate((poses, c2w[None, ...]))

                        file = parts[-1][:-1] #remove \n
                        print(file)
                        print(f'{dataset_path}/{dir}/{file}')
                        #Load images
                        img = cv2.imread(f'{dataset_path}/{dir}/{file}')[..., [2, 1, 0]] #bgr to rgb
                    
                        if images is None:
                            images = img[None, ...]
                        else:
                            images = np.concatenate((images, img[None, ...]))

                        i += 1
                        if i >= 7:
                            break
            
            # Load intrinsics
            avg_focal_length = 0.0
            with open(dataset_path+f'/cameras_{dir}.txt', 'r') as f:
                total = 0
                for i, line in enumerate(f):
                    if line[0]!='#':
                        parts = line.split(" ")
                        avg_focal_length += float(parts[-4])
                        print(float(line[-4]), parts[-4])
                    total = i+1
                
                avg_focal_length /= total

            num_images, height, width, num_channels = images.shape
            self.images = torch.from_numpy(images).to(device)
            self.poses = torch.from_numpy(poses).to(device)
            focal_length = torch.from_numpy(np.array([avg_focal_length])).to(device)
            print("Focal", focal_length)
            self.hwf = (height, width, focal_length)

    def load_meshroom(self, dataset_path, device=torch.device('cuda')):
        if dataset_path[-1] == '/':
            dataset_path = dataset_path[:-1]
        
        poses = None
        images = None
        subdirs = get_subdirs(dataset_path)
        
        for i_dir, dir in enumerate(subdirs):
            with open(f"{dataset_path}/cameras_{dir}.sfm") as json_file:
                cameras = json.load(json_file)

                for i, pose in tqdm(enumerate(cameras["poses"]), unit="pose", desc="Loading poses"):
                    # Load poses
                    pose_id = pose["poseId"]
                    print(pose['pose']['transform']['rotation'])
                    R = np.array(pose['pose']['transform']['rotation']).reshape((3, 3)).T.astype(float)
                    print(R)
                    C = np.array(pose['pose']['transform']['center']).reshape((3, 1)).astype(float)
                    pose = np.concatenate((R, -R@C), axis=1)

                    if poses is None:
                        poses = pose[None, ...]
                    else:
                        poses = np.concatenate((poses, pose[None, ...]))

                    # Load images
                    view = [view for view in cameras["views"] if view["poseId"] == pose_id][0]
                    intrinsicId = view["intrinsicId"]
                    path = view["path"]
                    file = path.split("/")[-1]

                    img = cv2.imread(f'{dataset_path}/{dir}/{file}')[..., [2, 1, 0]] #bgr to rgb
                
                    if images is None:
                        images = img[None, ...]
                    else:
                        images = np.concatenate((images, img[None, ...]))

                    if i>5:
                        break

                    # Load intrinsics
                    focal_length = float([intrinsic["pxFocalLength"] for intrinsic in cameras["intrinsics"] if intrinsic["intrinsicId"] == intrinsicId][0])
                    print("Focal_length", focal_length)


            self.subdirs_indices[i_dir] = list(range(self.subdirs_indices[i_dir][0], images.shape[0]))
            self.subdirs_indices.append([images.shape[0]])
            self.subdirs.append(dir)
        
        self.subdirs_indices = self.subdirs_indices[:-1]
        num_images, height, width, num_channels = images.shape
        
        self.images = torch.from_numpy(images).to(device)
        self.poses = torch.from_numpy(poses).to(device)
        focal_length = torch.from_numpy(np.array([focal_length])).to(device)
        print("Focal", focal_length)
        self.hwf = (height, width, focal_length)
        
    def get_val_pose_img_hwf(self, val_image=0):
        i_train, i_val, i_test = self.i_split
        return self.poses[i_val[val_image]], self.images[i_val[val_image]], self.hwf