import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from PIL import Image
import json

def get_validation_rays(pose_val, hwf, get_rays_origins_and_directions):
    height, width, focal_length = hwf
    
    rays_od = torch.cat(get_rays_origins_and_directions(height, width, focal_length, pose_val), dim=-1)

    view_dirs = rays_od[..., 3:]
    view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
    rays_odv = torch.cat([rays_od, view_dirs], dim=-1)

    rays_odv = torch.reshape(rays_odv, [-1, 9]).float()
    return rays_odv

def batchify(locations, view_dirs, chunksize):
    locs_flat = torch.reshape(locations, [-1, locations.shape[-1]])
    view_dirs = torch.broadcast_to(view_dirs, locations.shape)
    view_dirs_flat = torch.reshape(view_dirs, [-1, view_dirs.shape[-1]])
    for i in range(0, locs_flat.shape[0], chunksize):
        #upper = i+chunksize if i+chunksize < locs_flat.shape[0] else locs_flat.shape[0]
        #print(f"{i}:{i+chunksize}, {locs_flat[i:i+chunksize].shape}, {locs_flat.shape[0]}")
        yield (locs_flat[i:i+chunksize], view_dirs_flat[i:i+chunksize])

class NeRFSubDataset():
    def __init__(self, rays, images, height, width):
        self.dataset = TensorDataset(rays, images)
        self.dataloader = DataLoader(self.dataset, batch_size=height*width, shuffle=False)
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
                 dataset, 
                 path, 
                 get_rays_origins_and_directions, 
                 device=torch.device('cuda')):

        if dataset == "tiny_nerf":
            self.poses, self.images, self.hwf, self.i_split = self.load_tiny_nerf(path, device=device)
        else:
            self.poses, self.images, self.hwf, self.i_split = self.load_synthetic(path, device=device)
        
        self.device = device
        self._create(get_rays_origins_and_directions)
    
    def _create(self, get_rays_origins_and_directions):
        i_train, i_val, i_test = self.i_split
        height, width, focal_length = self.hwf
        focal_length = focal_length.to(self.poses)
        
        rays_od = [torch.cat(get_rays_origins_and_directions(height, width, focal_length, p), dim=-1) for p in self.poses]
        rays_od = torch.stack(rays_od)
        view_dirs = rays_od[..., 3:]
        view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
        rays_odv = torch.cat([rays_od, view_dirs], dim=-1)
        
        rays_train = torch.reshape(rays_odv[i_train], [-1, 9]).to(self.device)
        images_train = torch.reshape(self.images[i_train], [-1, 3]).to(self.device)
        rays_val = torch.reshape(rays_odv[i_val], [-1, 9]).to(self.device)
        images_val = torch.reshape(self.images[i_val], [-1, 3]).to(self.device)
        rays_test = torch.reshape(rays_odv[i_test], [-1, 9]).to(self.device)
        images_test = torch.reshape(self.images[i_test], [-1, 3]).to(self.device)
        focal_length = focal_length.to(self.device)
        
        self.dataset_train = NeRFSubDataset(rays_train, images_train, height, width)
        self.dataset_val = NeRFSubDataset(rays_val, images_val, height, width)
        self.dataset_test = NeRFSubDataset(rays_test, images_test, height, width)


    def next_batch(self, dataset="train"):
        dataset = self.dataset_train if dataset == "train" else (self.dataset_val if dataset == "val" else self.dataset_test)
        return dataset.next_batch()
    
    def load_synthetic(self, path="./synthetic/lego/", device=torch.device('cuda'), already_built=True, 
                       save_to="./synthetic/synthetic_lego.npz"):
        if already_built:
            data = np.load(path)
            images = data["images"]
            poses = data["poses"]
            focal_length = data["focal"]
            num_images, height, width, num_channels = images.shape
        
        else:
            dirs = ["train/", "val/", "test/"]

            poses = None
            images = None

            for dir in dirs:
                # Load poses
                with open(path+f'transforms_{dir[:-1]}.json') as json_file:
                    data = json.load(json_file)

                    camera_angle_x = data["camera_angle_x"]
                    for frame in data["frames"]:
                        pose = np.asarray(frame["transform_matrix"])
                        if poses is None:
                            poses = pose[None, ...]
                        else:
                            poses = np.concatenate((poses, pose[None, ...]))
                            
                        print(f"Loading pose {poses.shape[0]} of {dir}", end="\r")
                        
                    print()

                # Load images
                for img in os.listdir(path+dir):
                    if "depth" in img:
                        continue

                    im = np.asarray(Image.open(path+dir+img))

                    if images is None:
                        images = im[None, ...]
                    else:
                        images = np.concatenate((images, im[None, ...]))

                    print(f"Loading image {images.shape[0]} of {dir}", end="\r")
                    
                print()

            num_images, height, width, num_channels = images.shape
            focal_length = width/(2 * np.tan(camera_angle_x/2))

            with open(save_to, 'wb') as f:
                np.savez(f, images=images, poses=poses, focal=focal_length)
        
        images = torch.from_numpy(images)
        poses = torch.from_numpy(poses)

        hwf = (height, width, focal_length)
        i_split = (list(range(100)), list(range(100, 200)), list(range(200, 400)))
        return poses, images, hwf, i_split

    def load_tiny_nerf(self, path="tiny_nerf_data.npz", device=torch.device('cuda')):
        i_train, i_val, i_test = list(range(0, 100)), list(range(100, 101)), list(range(0, 1))
        i_split = (i_train, i_val, i_test)

        data = np.load(path)

        # Images - shape: (num_images, height, width, channels)
        images = data["images"]
        images = torch.from_numpy(images)
        num_images, height, width, num_channels = images.shape

        # Camera extrinsics
        poses = data["poses"]
        poses = torch.from_numpy(poses)

        # Focal length (intrinsics)
        focal_length = data["focal"]
        focal_length = torch.from_numpy(focal_length).to(device)
        
        hwf = (height, width, focal_length)

        return poses, images, hwf, i_split

    def get_val_pose_img_hwf(self, val_image=0):
        i_train, i_val, i_test = self.i_split
        return self.poses[i_val[val_image]], self.images[i_val[val_image]], self.hwf