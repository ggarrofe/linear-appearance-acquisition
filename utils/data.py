
import numpy as np
import os
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import cv2, imageio

from tqdm import tqdm
from utils.load_llff import load_llff_data
import utils.utils as utils
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import gc

def _minify(basedir, subdir=None, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, f'{subdir}_{r}')
        if not os.path.exists(imgdir):
            print("needtoload")
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, f'{subdir}_{r[1]}x{r[0]}')
        if not os.path.exists(imgdir):
            print("needtoload")
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = basedir if subdir is None else os.path.join(basedir, subdir)
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()
    if subdir is None:
        subdir = 'images'

    for r in factors + resolutions:
        if isinstance(r, int):
            name = '{}_{}'.format(subdir, r)
            resizearg = '{}%'.format(100./r)
        else:
            name = '{}_{}x{}'.format(subdir, r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, imgdir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
 

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
        for name in os.listdir(path):
            if os.path.isdir(os.path.join(path, name)) == False:
                continue
            if name[-1].isdigit() or ".tmp" in name or "_xmp" in name or "relighting_ommit" in name:
                continue
            subdirs.append(name)

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
    return images_size, num_images_list

class NeRFSubDataset():
    def __init__(self, rays, images, hwf, name, batch_size=None, shuffle=False, light_poses=None):
        print(f"\tCreating {name} dataset with rays ({rays.shape}) and images ({images.shape}) - shuffle {shuffle}")
        rays[..., 3:6] /= torch.norm(rays[..., 3:6])
        self.dataset = TensorDataset(rays, images)
        self.name = name
        self.hwf = hwf
        self.light_poses = light_poses

        height, width, focal = hwf
        self.n_images = int(images.shape[0]/(height*width))

        if batch_size is None:
            batch_size=height*width

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.iterator = iter(self.dataloader)

        self.inf_value = -1.0
        self.light_rays = None
        self.points = None
        self.half_angles = None

    def compute_depths(self, scene, device=torch.device("cpu")):
        h, w, f = self.hwf
        self.depths = torch.zeros((self.dataset.tensors[0].shape[0])).to(device)
        self.points = torch.zeros((self.dataset.tensors[0].shape[0], 3)).to(device)
        
        for i in tqdm(range(0, self.dataset.tensors[0].shape[0], h*w), desc="Computing depths", leave=False):
            rays_od = self.dataset.tensors[0][i:i+h*w, :6]
            
            hit = utils.cast_rays(scene, rays_od)
            depths = torch.from_numpy(hit).to(device)
            hit = hit.reshape(h, w).T.flatten()

            # depths are inf if the ray does not hit the mesh
            self.depths[i:i+h*w] = torch.from_numpy(hit).to(device)
            points = rays_od[..., :3] + rays_od[..., 3:] * depths[..., None] 
            points = points.reshape(h, w, points.shape[-1])
            self.points[i:i+h*w] = torch.transpose(points, 0, 1).flatten(end_dim=1).to(device)

        self.points = torch.nan_to_num(self.points, posinf=self.inf_value, neginf=self.inf_value, nan=-1.0)

    def compute_normals(self):
        h, w, f = self.hwf
        self.normals = torch.zeros_like(self.points, requires_grad=False, device=self.depths.device)
            
        padding_h = torch.zeros((h,1)).to(self.depths)
        padding_w = torch.zeros((1,w)).to(self.depths)
        
        for i in range(0, self.dataset.tensors[0].shape[0], h*w):
            depths = self.depths[i:i+h*w].reshape(h, w)
            dzdx = torch.cat([padding_w, (depths[2:,...] - depths[:-2,...])/2, padding_w], dim=0)
            dzdy = torch.cat([padding_h, (depths[..., 2:] - depths[..., :-2])/2, padding_h], dim=1)
            ones = torch.ones_like(dzdx)
            direction = torch.stack([-dzdx, -dzdy, ones], dim=-1)

            magnitude = torch.sqrt(torch.pow(direction[..., 0], 2)+torch.pow(direction[..., 1], 2)+torch.pow(direction[..., 2], 2)).unsqueeze(-1)
            normals = direction/magnitude * 0.5 + 0.5
            self.normals[i:i+h*w] = torch.nan_to_num(normals, nan=-1.0).reshape((h*w, 3))

    def compute_halfangles(self):
        assert self.light_poses is not None, "Light poses info is missing"
        
        light_pos = torch.stack([c2w[:3, 3][None, None, ...].expand((self.hwf[0], self.hwf[1], 3)) for c2w in self.light_poses])
        light_pos = light_pos.reshape((-1, 3))

        cam_pos = self.dataset.tensors[0][..., :3]

        V = (cam_pos - self.points).float()
        self.L = (light_pos - self.points).float()
        self.half_angles = F.normalize((V + self.L), dim=1).float()

        self.NdotH = torch.bmm(self.normals.unsqueeze(1), self.half_angles.unsqueeze(2)).squeeze(2)
        self.NdotL = torch.bmm(self.normals.unsqueeze(1), self.L.unsqueeze(2)).squeeze(2)

        self.light_rays = torch.stack([torch.cat(NeRFDataset.get_rays_origins_and_directions(c2w, self.hwf), dim=-1) 
                                        for c2w in tqdm(self.light_poses, unit="pose", desc="Generating ray lights")])
        self.light_rays = self.light_rays.reshape((-1, 6))

    def switch_2_xnv_dataset(self, device=torch.device("cpu")):
        viewdirs = self.dataset.tensors[0][..., 6:].to(device)
        X = torch.cat([self.points, self.normals, viewdirs], dim=-1)
        shape_1 = self.dataset.tensors[1]
        del self.dataset
        del self.dataloader
        del self.iterator
        gc.collect()
        self.dataset = TensorDataset(X, shape_1)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.iterator = iter(self.dataloader)

    def switch_2_xv_dataset(self, device=torch.device("cpu")):
        viewdirs = self.dataset.tensors[0][..., 6:].to(device)
        X = torch.cat([self.points, viewdirs], dim=-1)
        shape_1 = self.dataset.tensors[1]
        del self.dataset
        del self.dataloader
        del self.iterator
        gc.collect()
        
        self.dataset = TensorDataset(X, shape_1)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.iterator = iter(self.dataloader)
    
    def switch_2_xn_dataset(self, device=torch.device("cpu")):
        X = torch.cat([self.points, self.normals], dim=-1)
        shape_1 = self.dataset.tensors[1]
        del self.dataset
        del self.dataloader
        del self.iterator
        gc.collect()
        
        self.dataset = TensorDataset(X, shape_1)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.iterator = iter(self.dataloader)

    def switch_2_X_NdotL_NdotH_dataset(self, device=torch.device("cpu")):
        del self.dataset
        del self.dataloader
        del self.iterator
        gc.collect()
        X_NdotL_NdotH, rgb = self.get_X_NdotL_NdotH_rgb(i=-1, device=device)
        self.dataset = TensorDataset(X_NdotL_NdotH, rgb)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.iterator = iter(self.dataloader)

    def switch_2_X_H_dataset(self, device=torch.device("cpu")):
        del self.dataloader
        del self.iterator
        gc.collect()
        X_NdotL_NdotH, rgb = self.get_X_H_rgb(i=-1, device=device)
        del self.dataset
        gc.collect()
        self.dataset = TensorDataset(X_NdotL_NdotH, rgb)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.iterator = iter(self.dataloader)

    def sort_clusters(self, clusters_id):
        order = torch.argsort(clusters_id)
        self.sorted_dataset = TensorDataset(self.dataset.tensors[0][order], self.dataset.tensors[1][order])
        self.clusters_id = clusters_id[order]
        self.dataloader = DataLoader(self.sorted_dataset, batch_size=self.batch_size)
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
        return self.dataset.tensors[0][i*h*w:(i+1)*h*w, :6].float().to(device)

    def get_xh(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]

        if self.half_angles is None and self.light_poses is not None:
            self.compute_halfangles()

        return self.half_angles[i*h*w:(i+1)*h*w].float().to(device)

    def get_light_rays(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]

        if self.light_rays is None and self.light_poses is not None:
            self.compute_halfangles()

        return self.light_rays[i*h*w:(i+1)*h*w].float().to(device)

    def get_X_NdotL_NdotH_rgb(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]

        if self.half_angles is None and self.light_poses is not None:
            self.compute_halfangles()

        if i >= 0:
            return torch.cat([self.get_points(i, device), 
                              self.NdotL[i*h*w:(i+1)*h*w].float().to(device),
                              self.NdotH[i*h*w:(i+1)*h*w].float().to(device)], dim=-1), \
                   self.dataset.tensors[1][i*h*w:(i+1)*h*w].float().to(device)

        else:
            return torch.cat([self.points, self.NdotL, self.NdotH], dim=-1).float().to(device), \
                   self.dataset.tensors[1].float().to(device)

    def get_X_L_N_H_rgb(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]

        if self.half_angles is None and self.light_poses is not None:
            self.compute_halfangles()

        if i >= 0:
            return torch.cat([self.get_points(i, device), 
                              self.L[i*h*w:(i+1)*h*w].float().to(device),
                              self.normals[i*h*w:(i+1)*h*w].float().to(device),
                              self.half_angles[i*h*w:(i+1)*h*w].float().to(device)], dim=-1), \
                   self.dataset.tensors[1][i*h*w:(i+1)*h*w].float().to(device)

        else:
            return torch.cat([self.points, self.L, self.normals, self.half_angles], dim=-1).float().to(device), \
                   self.dataset.tensors[1].float().to(device)
    
    def get_X_N_H_rgb(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]

        if self.half_angles is None and self.light_poses is not None:
            self.compute_halfangles()

        if i >= 0:
            return torch.cat([self.get_points(i, device), 
                              self.normals[i*h*w:(i+1)*h*w].float().to(device),
                              self.half_angles[i*h*w:(i+1)*h*w].float().to(device)], dim=-1), \
                   self.dataset.tensors[1][i*h*w:(i+1)*h*w].float().to(device)

        else:
            return torch.cat([self.points, self.normals, self.half_angles], dim=-1).float().to(device), \
                   self.dataset.tensors[1].float().to(device)
    
    def get_X_H_rgb(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]

        if self.half_angles is None and self.light_poses is not None:
            self.compute_halfangles()

        if i >= 0:
            return torch.cat([self.get_points(i, device), 
                              self.half_angles[i*h*w:(i+1)*h*w].float().to(device)], dim=-1), \
                   self.dataset.tensors[1][i*h*w:(i+1)*h*w].float().to(device)

        else:
            return torch.cat([self.points, self.half_angles], dim=-1).float().to(device), \
                   self.dataset.tensors[1].float().to(device)

    def get_X_NdotL_NdotH_rgb_shape(self):
        return (self.dataset.tensors[0].shape[0], 5)

    def get_xh_rgb(self, i, device=torch.device('cuda')):
        h = self.hwf[0]
        w = self.hwf[1]

        if self.half_angles is None and self.light_poses is not None:
            self.compute_halfangles()

        if i >= 0:
            return torch.cat([self.get_points(i, device),
                            self.half_angles[i*h*w:(i+1)*h*w].float().to(device)], dim=-1), \
                   self.dataset.tensors[1][i*h*w:(i+1)*h*w].float().to(device)

        else:
            return torch.cat([self.points, self.half_angles], dim=-1).float().to(device), \
                   self.dataset.tensors[1].float().to(device)

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
        
    def get_tensors(self, device=torch.device('cuda')):
        return self.dataset.tensors[0].float().to(device), self.dataset.tensors[1].float().to(device), self.depths

    def get_sorted_tensors(self, device=torch.device('cuda')):
        return self.sorted_dataset.tensors[0].float().to(device), self.sorted_dataset.tensors[1].float().to(device)

class NeRFDataset():

    def __init__(self, 
                 args):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.dataset_to_gpu else torch.device("cpu")
        self.subdirs_indices = []
        self.subdirs = []
        self.light_poses = None

        if args.dataset_type == "tiny":
            self.load_tiny(args.dataset_path, device=self.device)
        elif args.dataset_type == "synthetic":
            self.load_synthetic(args.dataset_path, 
                                device=self.device, 
                                factor=args.factor,
                                train_images=args.train_images, 
                                val_images=args.val_images,
                                test_images=args.test_images,
                                load_light=args.load_light)
        elif args.dataset_type == "llff":
            self.load_llff(args.dataset_path, 
                           device=self.device, 
                           factor=args.factor, 
                           train_images=args.train_images, 
                           val_images=args.val_images,
                           test_images=args.test_images)
        elif args.dataset_type == "meshroom":
            self.load_meshroom(args.dataset_path, device=self.device)
        elif args.dataset_type == "colmap":
            self.load_colmap(args.dataset_path, device=self.device)
        
        self.args = args
        self._create()
    
    def _create(self):
        #focal_length = focal_length.to(self.poses)
        
        rays_od = [torch.cat(self.get_rays_origins_and_directions(c2w, self.hwf), dim=-1) for c2w in tqdm(self.poses, unit="pose", desc="Generating rays")]
        rays_od = torch.stack(rays_od)

        print("Creating datasets...")
        self.subdatasets = []

        for indices, name in zip(self.subdirs_indices, self.subdirs):
            rays = torch.reshape(rays_od[indices], [-1, 6])
            images = torch.reshape(self.images[indices], [-1, 3])
            light_poses = None if self.light_poses is None else self.light_poses[indices]

            print(f"\tComputing view dirs for {name}...")
            view_dirs = rays[..., 3:]
            view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
            rays_odv = torch.cat([rays, view_dirs], dim=-1)

            self.subdatasets.append(NeRFSubDataset(rays_odv, 
                                                   images, 
                                                   self.hwf, 
                                                   name, 
                                                   batch_size=min(self.args.batch_size, rays_odv.shape[0]), 
                                                   shuffle=self.args.shuffle, 
                                                   light_poses=light_poses))
        
        # Trying to free some memory to deal with big datasets
        delattr(self, "images")

        print("Datasets created successfully\n")

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

    def get_rays_od(self, dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_rays_od(img, device=device)

    def get_VLH(self, dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_VLH(img, device=device)

    def get_xh_rgb(self, dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_xh_rgb(img, device=device)
    
    def get_X_NdotL_NdotH_rgb(self, dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_X_NdotL_NdotH_rgb(img, device=device)

    def get_X_NdotL_NdotH_rgb_shape(self, dataset="train"):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_X_NdotL_NdotH_rgb_shape()

    def get_X_L_N_H_rgb(self, dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_X_L_N_H_rgb(img, device=device)

    def get_X_N_H_rgb(self, dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_X_N_H_rgb(img, device=device)

    def get_X_H_rgb(self, dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_X_H_rgb(img, device=device)

    def get_light_rays(self, dataset="train", img=0, device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_light_rays(img, device=device)

    def get_n_images(self, dataset="train"):
        subdataset = [d for d in self.subdatasets if d.name == dataset]
        if len(subdataset) == 0: return 0
        return subdataset[0].n_images

    def compute_depths(self, device=torch.device('cuda')):
        mesh = o3d.io.read_triangle_mesh(self.args.mesh_path, print_progress=True)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        
        for subdataset in self.subdatasets:
            subdataset.compute_depths(scene, device=device)

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

    def get_tensors(self, dataset="train", device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_tensors(device=device)

    def get_sorted_tensors(self, dataset="train", device=torch.device('cuda')):
        subdataset = [d for d in self.subdatasets if d.name == dataset][0]
        return subdataset.get_sorted_tensors(device=device)

    def switch_2_xnv_dataset(self, device=torch.device('cpu')):
        self.compute_depths(device=device)
        self.compute_normals()

        for d in self.subdatasets:
            d.switch_2_xnv_dataset(device=device)

    def switch_2_xv_dataset(self, device=torch.device('cpu')):
        self.compute_depths(device=device)

        for d in self.subdatasets:
            d.switch_2_xv_dataset(device=device)

    def switch_2_xn_dataset(self, device=torch.device('cpu')):
        self.compute_depths(device=device)
        self.compute_normals()

        for d in self.subdatasets:
            d.switch_2_xn_dataset(device=device)

    def switch_2_X_NdotL_NdotH_dataset(self, device=torch.device('cpu')):
        for d in self.subdatasets:
            d.switch_2_X_NdotL_NdotH_dataset(device=device)

    def switch_2_X_H_dataset(self, device=torch.device('cpu')):
        for d in self.subdatasets:
            d.switch_2_X_H_dataset(device=device)

    def compute_halfangles(self):
        for d in self.subdatasets:
            d.compute_halfangles()

    def sort_clusters(self, cluster_ids):
        subdataset = [d for d in self.subdatasets if d.name == "train"][0]
        subdataset.sort_clusters(cluster_ids)

    def imread(self, f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)[...,:3]/255.
        else:
            return imageio.imread(f)/255.

    def load_synthetic(self,  
                  basedir,
                  device=torch.device('cuda'), 
                  factor=None, 
                  train_images=100, 
                  val_images=100, 
                  test_images=100,
                  load_light=False):

        if basedir[-1] == '/':
            basedir = basedir[:-1]
        
        subdirs = get_subdirs(basedir)
        print(subdirs)
        sfx=''
        if factor is not None and factor > 1:
            sfx = '_{}'.format(factor)
            factor = factor
            for subdir in subdirs:
                _minify(basedir, subdir=subdir, factors=[factor])
            
        imgs_size, num_images = get_images_size(basedir, subdirs, factor=factor)
        if "train" in subdirs: num_images[subdirs.index("train")] = train_images
        if "val" in subdirs: num_images[subdirs.index("val")] = val_images
        if "test" in subdirs: num_images[subdirs.index("test")] = test_images
        imgs_size[0] = np.array(num_images).sum()
        images = np.zeros(imgs_size)
        poses = np.zeros((imgs_size[0], 3, 4))
        if load_light:
            light_poses = np.zeros((imgs_size[0], 3, 4))
        i_imgs = 0
        
        for i_dir, subdir in enumerate(subdirs):
            if num_images[i_dir] == 0: continue
            imgdir = os.path.join(basedir, subdir + sfx)
    
            if not os.path.exists(imgdir):
                print( imgdir, 'does not exist, returning' )
                return

            poses_old = poses.copy()

            if load_light:
                print(f"Loading {subdir} - {num_images[i_dir]} poses, images ({imgs_size[1]}x{imgs_size[2]}) and lights...")
            else:
                print(f"Loading {subdir} - {num_images[i_dir]} poses and images ({imgs_size[1]}x{imgs_size[2]})...")

            with open(f"{basedir}/transforms_{subdir}.json") as json_file:
                data = json.load(json_file)
                camera_angle_x = data["camera_angle_x"]

                for i, frame in tqdm(enumerate(data["frames"][:num_images[i_dir]]), unit="frame", leave=False, desc="Loading frames" if len(subdirs)==1 else f"Loading {subdir} frames"):
                    # Load poses
                    poses[i_imgs+i:i_imgs+i+1] = np.asarray(frame["transform_matrix"])[None, :3, ...]

                    # Load images
                    file = frame["file_path"].split('/')[-1]
                    images[i_imgs+i:i_imgs+i+1] = self.imread(f'{imgdir}/{file}.png')[None, ...] 

                    # Load light sources
                    if load_light:
                        light_poses[i_imgs+i:i_imgs+i+1] = np.asarray(frame["light_transform_matrix"])[None, :3, ...]

            utils.summarize_diff(poses_old, poses)

            self.subdirs_indices.append(list(range(i_imgs, i_imgs+num_images[i_dir])))
            self.subdirs.append(subdir)
            i_imgs += num_images[i_dir]

        num_images, height, width, num_channels = images.shape
        focal_length = width/(2 * np.tan(camera_angle_x/2))
        focal_length = torch.from_numpy(np.array([focal_length])).to(device)
        self.images = torch.from_numpy(images).to(device)
        self.poses = torch.from_numpy(poses).to(device)
        self.hwf = (height, width, focal_length)

        if load_light:
            self.light_poses = torch.from_numpy(light_poses).to(device)

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
    
    def load_llff(self, dataset_path, device=torch.device('cuda'), factor=1, train_images=100, val_images=100, test_images=100):
        subdirs = get_subdirs(dataset_path)
        print(subdirs)

        imgs_size, num_images = get_images_size(dataset_path, subdirs, factor=factor)
        if "train" in subdirs: num_images[subdirs.index("train")] = train_images
        if "val" in subdirs: num_images[subdirs.index("val")] = val_images
        if "test" in subdirs: num_images[subdirs.index("test")] = test_images
        imgs_size[0] = np.array(num_images).sum()
        images = np.zeros(imgs_size)
        poses = np.zeros((imgs_size[0], 3, 4))
        i_imgs = 0

        for i_dir, dir in enumerate(subdirs):
            if num_images[i_dir] == 0: continue
            poses_old = poses.copy()
            print(f"Loading {dir} - {num_images[i_dir]} images...")
            
            hwf, poses[i_imgs:i_imgs+num_images[i_dir]] = load_llff_data(dataset_path, 
                                 poses[i_imgs:i_imgs+num_images[i_dir]],
                                 images[i_imgs:i_imgs+num_images[i_dir]],
                                 factor=factor, 
                                 subdir=dir,
                                 i=i_imgs,
                                 i_n=i_imgs+num_images[i_dir],
                                 n_imgs=num_images[i_dir])
            
            utils.summarize_diff(poses_old, poses)

            self.subdirs_indices.append(list(range(i_imgs, i_imgs+num_images[i_dir])))
            self.subdirs.append(dir)
            i_imgs += num_images[i_dir]

        self.images = torch.from_numpy(images).to(device)
        self.poses = torch.from_numpy(poses).to(device)
        focal_length = torch.from_numpy(np.array([hwf[2]])).to(device)
        self.hwf = (int(hwf[0]), int(hwf[1]), focal_length)

    def load_reality_capture(self, dataset_path, device=torch.device('cuda'), factor=1, train_images=100, val_images=100, test_images=100):
        subdirs = get_subdirs(dataset_path)
        print(subdirs)

        imgs_size, num_images = get_images_size(dataset_path, subdirs, factor=factor)
        if "train" in subdirs: num_images[subdirs.index("train")] = train_images
        if "val" in subdirs: num_images[subdirs.index("val")] = val_images
        if "test" in subdirs: num_images[subdirs.index("test")] = test_images
        imgs_size[0] = np.array(num_images).sum()
        images = np.zeros(imgs_size)
        poses = np.zeros((imgs_size[0], 3, 4))
        i_imgs = 0

        for i_dir, dir in enumerate(subdirs):
            poses_old = poses.copy()
            print(f"Loading {dir} - {num_images[i_dir]} images...")

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
        

