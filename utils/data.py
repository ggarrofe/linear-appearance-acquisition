import numpy as np
import os
import json
import torch
from torch.utils.data import TensorDataset, DataLoader
import cv2, imageio

from tqdm import tqdm
from utils.load_llff import load_llff_data
import utils.utils as utils
from scipy.spatial.transform import Rotation as R

from pprint import pprint

def get_subdirs(path) -> list:
    """ Check if the provided path contains the images or has other inner directories that 
        contain the images. Ommits the resized folders that end with a number.

        e.g.: dataset           Has the following subdirs: ["train", "test", "val"]
                |-> train
                |-> test
                |-> val
                |-> train_1

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
        subdirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and not name[-1].isdigit() and not ".tmp" in name]

    if len(subdirs) == 0:
        subdirs = ['.'] 
    
    return subdirs

def get_images_size(basedir, subdirs, factor=1):
    num_images = 0
    num_images_list = []

    for subdir in subdirs:
        files_list = [f for f in os.listdir(os.path.join(basedir, subdir)) 
                                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        num_images += len(files_list)
        num_images_list.append(len(files_list))
        
    img_size = list(imageio.imread(os.path.join(basedir, subdir, files_list[0])).shape) 
    img_size = [int(s/factor) for s in img_size]
    img_size[-1] = 3
    images_size = [num_images]
    images_size.extend(img_size)
    return tuple(images_size), num_images_list

class NeRFSubDataset():
    def __init__(self, rays, images, hwf, name, batch_size=None, shuffle=False):
        print(f"\tCreating {name} dataset with rays ({rays.shape}) and images ({images.shape})")
        rays[..., 3:6] /= torch.norm(rays[..., 3:6])
        self.dataset = TensorDataset(rays, images)
        self.name = name
        self.hwf = hwf
        height, width, focal = hwf
        
        self.n_images = int(images.shape[0]/(height*width))

        if batch_size is None:
            batch_size=height*width

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.iterator = iter(self.dataloader)

        self.inf_value = 10.0

    def compute_depths(self, scene):
        h, w, f = self.hwf
        self.depths = torch.zeros((self.dataset.tensors[0].shape[0]))
        self.points = torch.zeros((self.dataset.tensors[0].shape[0], 3))
        
        for i in range(0, self.dataset.tensors[0].shape[0], h*w):
            rays_od = self.dataset.tensors[0][i:i+h*w, :6]
            #rays_od[..., 3:] /= torch.norm(rays_od[..., 3:])
            norm=torch.norm(rays_od[..., 3:])
            #print("norms shape depths", norm.shape, rays_od[..., :3].shape)
            #print("norms depth", norm)
            hit = utils.cast_rays(scene, rays_od)
            depths = torch.from_numpy(hit)
            hit = hit.reshape(h, w).T.flatten()

            # depths are inf if the ray does not hit the mesh
            self.depths[i:i+h*w] = torch.from_numpy(hit) #* torch.norm(rays_od[..., 3:], dim=-1)
            points = rays_od[..., :3] + rays_od[..., 3:] * depths[..., None] 
            points = points.reshape(h, w, points.shape[-1])
            self.points[i:i+h*w] = torch.transpose(points, 0, 1).flatten(end_dim=1)

        self.points = torch.nan_to_num(self.points, posinf=self.inf_value, neginf=self.inf_value, nan=0.0)

    def compute_normals(self):
        """
        Consider your range image is a function z(x,y).
        The normal to the surface is in the direction (-dz/dx,-dz/dy,1). (Where by dz/dx I mean 
        the differential: the rate of change of z with x). And then normals are conventionally 
        normalized to unit length.

        Incidentally, if you're wondering where that (-dz/dx,-dz/dy,1) comes from... if you take 
        the 2 orthogonal tangent vectors in the plane parellel to the x and y axes, those are (1,0,dzdx) 
        and (0,1,dzdy). The normal is perpendicular to the tangents, so should be (1,0,dzdx)X(0,1,dzdy) 
        - where 'X' is cross-product - which is (-dzdx,-dzdy,1). So there's your cross product derived 
        normal, but there's little need to compute it so explicitly in code when you can just use the 
        resulting expression for the normal directly.
        """

        h, w, f = self.hwf
        self.normals = torch.zeros_like(self.points, requires_grad=False, device=self.depths.device)
            
        padding_h = torch.zeros((h,1))
        padding_w = torch.zeros((1,w))
        
        for i in range(0, self.dataset.tensors[0].shape[0], h*w):
            depths = self.depths[i:i+h*w].reshape(h, w)
            dzdx = torch.cat([padding_w, (depths[2:,...] - depths[:-2,...])/2, padding_w], dim=0)
            dzdy = torch.cat([padding_h, (depths[..., 2:] - depths[..., :-2])/2, padding_h], dim=1)
            ones = torch.ones_like(dzdx)
            direction = torch.stack([-dzdx, -dzdy, ones], dim=-1)

            magnitude = torch.sqrt(torch.pow(direction[..., 0], 2)+torch.pow(direction[..., 1], 2)+torch.pow(direction[..., 2], 2)).unsqueeze(-1)
            normals = direction/magnitude * 0.5 + 0.5
            self.normals[i:i+h*w] = torch.nan_to_num(normals).reshape((h*w, 3))

    def create_xnv_dataset(self, scene):
        self.compute_depths(scene)
        self.compute_normals()

        viewdirs = self.dataset.tensors[0][..., 6:]
        X = torch.cat([self.points, self.normals, viewdirs], dim=-1)
        
        self.dataset = TensorDataset(X, self.dataset.tensors[1])
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.iterator = iter(self.dataloader)

    def next_batch(self, device=torch.device('cuda')):
        try:
            batch_rays, target_rgb = next(self.iterator)
            
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch_rays, target_rgb = next(self.iterator)

        return batch_rays.float().to(device), target_rgb.float().to(device)

    def get_image(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]
        
        imgs = self.dataset.tensors[1][i*h*w:(i+1)*h*w, ...].float().to(device)
        return imgs

    def get_X_target(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]
        X = self.dataset.tensors[0][i*h*w:(i+1)*h*w, ...].float().to(device)
        target = self.dataset.tensors[1][i*h*w:(i+1)*h*w, ...].float().to(device)
        return X, target, self.depths[i*h*w:(i+1)*h*w]

    def get_rays_od(self, i, device):
        h = self.hwf[0]
        w = self.hwf[1]
        return self.dataset.tensors[0][i*h*w:(i+1)*h*w, ..., :6].float().to(device)

    def get_depths(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]
        return self.depths[i*h*w:(i+1)*h*w].to(device)

    def get_normals(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]
        return self.normals[i*h*w:(i+1)*h*w].to(device)

    def get_points(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]
        return self.points[i*h*w:(i+1)*h*w].to(device)
        
class NeRFDataset():

    def __init__(self, 
                 args,
                 device=torch.device('cuda')):

        self.device = device
        self.subdirs_indices = []
        self.subdirs = []

        if args.dataset_type == "tiny":
            self.load_tiny(args.dataset_path, device=device)
        elif args.dataset_type == "synthetic":
            self.load_synthetic(args.dataset_path, device=device)
        elif args.dataset_type == "llff":
            self.load_llff(args.dataset_path, device=device, factor=args.factor)
        elif args.dataset_type == "meshroom":
            self.load_meshroom(args.dataset_path, device=device)
        elif args.dataset_type == "colmap":
            self.load_colmap(args.dataset_path, device=device)
        
        self._create(args.batch_size, args.test, args.shuffle)
    
    def _create(self, batch_size=None, test=False, shuffle=False):
        #focal_length = focal_length.to(self.poses)
        
        rays_od = [torch.cat(self.get_rays_origins_and_directions(c2w, self.hwf), dim=-1) for c2w in tqdm(self.poses, unit="pose", desc="Generating rays")]
        rays_od = torch.stack(rays_od)
        
        print("Creating datasets...")
        self.subdatasets = []

        for indices, name in zip(self.subdirs_indices, self.subdirs):
            rays = torch.reshape(rays_od[indices], [-1, 6])
            images = torch.reshape(self.images[indices], [-1, 3])
            if test:
                h = self.hwf[0]
                w = self.hwf[1]
                rays = rays[:10*h*w]
                images = images[:10*h*w]

            print(f"\tComputing view dirs for {name}...")
            view_dirs = rays[..., 3:]
            view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
            rays_odv = torch.cat([rays, view_dirs], dim=-1)

            self.subdatasets.append(NeRFSubDataset(rays_odv, images, self.hwf, name, batch_size=batch_size, shuffle=shuffle))

        print("Datasets created successfully\n")
        #focal_length = focal_length.to(self.device)

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
        
        rays_d = torch.sum(directions[..., None, :] * R, dim=-1)
        #print("raus shape", rays_d.shape)
        norm = torch.norm(rays_d, dim=-1)
        #print("norm shape data", norm[..., None].shape)
        rays_d /= norm[..., None]
        
        rays_o = t.expand(rays_d.shape)
        
        return rays_o, rays_d

    def next_batch(self, dataset="train", device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.next_batch(device)
    
    def get_image(self, dataset="val", i=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_image(i, device=device)

    def get_X_target(self, dataset="val", i=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_X_target(i, device=device)

    def get_rays_od(self,dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_rays_od(img, device=device)

    def get_n_images(self, dataset="train"):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.n_images

    def compute_depths(self, scene):
        for subdataset in self.subdatasets:
            subdataset.compute_depths(scene)

    def get_depths(self, dataset="val", i=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_depths(i, device=device)

    def compute_normals(self):
        for subdataset in self.subdatasets:
            subdataset.compute_normals()

    def get_normals(self, dataset="val", i=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_normals(i, device=device)

    def get_points(self, dataset="val", i=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_points(i, device=device)

    def create_xnv_dataset(self, scene):
        for d in self.subdatasets:
            d.create_xnv_dataset(scene)

    def load_synthetic(self,  
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
       
    def load_llff(self, dataset_path, device=torch.device('cuda'), factor=1):
        subdirs = get_subdirs(dataset_path)
        print(subdirs)

        imgs_size, num_images = get_images_size(dataset_path, subdirs, factor=factor)
        images = np.zeros(imgs_size)
        poses = np.zeros((imgs_size[0], 3, 4))
        i_imgs = 0

        for i_dir, dir in enumerate(subdirs):
            poses_old = poses.copy()
            print(f"Loading {dir} - {num_images[i_dir]} images...")
            
            hwf, poses[i_imgs:i_imgs+num_images[i_dir]] = load_llff_data(dataset_path, 
                                 poses[i_imgs:i_imgs+num_images[i_dir]],
                                 images[i_imgs:i_imgs+num_images[i_dir]],
                                 factor=factor, 
                                 subdir=dir,
                                 i=i_imgs,
                                 i_n=i_imgs+num_images[i_dir])
            
            utils.summarize_diff(poses_old, poses)

            self.subdirs_indices.append(list(range(i_imgs, i_imgs+num_images[i_dir])))
            self.subdirs.append(dir)
            i_imgs += num_images[i_dir]

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