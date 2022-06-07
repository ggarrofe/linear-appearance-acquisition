import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os
from PIL import Image
import json


def load_synthetic(path="./synthetic/lego/", device=torch.device('cuda'), already_built=True, save_to="./synthetic/synthetic_lego.npz"):
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

        lengths = {}

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


def load_tiny_nerf(path="tiny_nerf_data.npz", device=torch.device('cuda')):
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


def create_dataset(poses, images, hwf, i_split, get_rays_origins_and_directions, batch_size=1024, device=torch.device('cuda')):
    i_train, i_val, i_test = i_split

    height, width, focal_length = hwf
    focal_length = focal_length.to(poses)
    
    rays_od = [torch.cat(get_rays_origins_and_directions(height, width, focal_length, p), dim=-1) for p in poses]
    rays_od = torch.stack(rays_od)
    view_dirs = rays_od[..., 3:]
    view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
    rays_odv = torch.cat([rays_od, view_dirs], dim=-1)
    
    rays_train = torch.reshape(rays_odv[i_train], [-1, 9]).to(device)
    images_train = torch.reshape(images[i_train], [-1, 3]).to(device)
    rays_val = torch.reshape(rays_odv[i_val], [-1, 9]).to(device)
    images_val = torch.reshape(images[i_val], [-1, 3]).to(device)
    rays_test = torch.reshape(rays_odv[i_test], [-1, 9]).to(device)
    images_test = torch.reshape(images[i_test], [-1, 3]).to(device)
    focal_length = focal_length.to(device)
    
    dataset_train = TensorDataset(rays_train, images_train)
    dataset_val = TensorDataset(rays_val, images_val)
    dataset_test = TensorDataset(rays_test, images_test)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    return loader_train, loader_val, loader_test

def get_validation_rays(pose_val, hwf, get_rays_origins_and_directions):
    height, width, focal_length = hwf
    
    rays_od = torch.cat(get_rays_origins_and_directions(height, width, focal_length, pose_val), dim=-1)

    view_dirs = rays_od[..., 3:]
    view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
    rays_odv = torch.cat([rays_od, view_dirs], dim=-1)

    rays_odv = torch.reshape(rays_odv, [-1, 9]).float()
    return rays_odv

def main():
    poses, images, hwf, i_split = load_synthetic(already_built=False)
    print(poses.shape, images.shape, hwf)

if __name__ == '__main__':
    main()