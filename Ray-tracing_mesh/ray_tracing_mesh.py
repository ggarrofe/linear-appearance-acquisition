import configargparse
import torch
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d
import open3d.core as o3c

import visualization as v

import sys
sys.path.append('../')
import utils.data as data

def load_mesh(path):
    return o3d.io.read_triangle_mesh(path, print_progress=True)

def torch2open3d(torch_tensor):
    return o3c.Tensor(torch_tensor.numpy())

def cast_rays_open3d(mesh, rays_od, hwf):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    ans = scene.cast_rays(torch2open3d(rays_od.float()))
    hit = ans['t_hit'].numpy().reshape(hwf[0], hwf[1])
    return hit

def parse_args():
    parser = configargparse.ArgumentParser(description="Initializes the geometry with a given mesh")
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--mesh', type=str, help='initial mesh path')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--test_path', type=str, help='path to test files')
    parser.add_argument('--dataset_type', type=str, help='type of dataset', choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--spherify', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument("--test", action='store_true', help='use reduced number of images')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # python3 ray_tracing_mesh.py -c ./configs/lego.conf
    args = parse_args()
    
    device = "cpu" #torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    mesh = load_mesh(args.mesh)
    dataset = data.NeRFDataset(args, device=device)

    for i in range(dataset.get_n_images()):
        rays_od = dataset.get_rays_od(i, device=device)

        if args.test_path is None:
            v.plot_rays_and_mesh(rays_od, mesh, dataset.hwf)
            
        else:
            print("[TEST] Drawing rays for loaded poses in red:")
            print(dataset.poses[i])
            poses_colmap, poses_llff = dataset.get_test_poses(i)
            print("[TEST] Drawing rays for poses LLFF in green:")
            print(poses_llff, end='\n\n')

            rays_colmap = torch.cat(data.NeRFDataset.get_rays_origins_and_directions(poses_colmap, dataset.hwf), dim=-1)
            rays_colmap = torch.reshape(rays_colmap, [-1, 6])
            rays_llff = torch.cat(data.NeRFDataset.get_rays_origins_and_directions(poses_llff, dataset.hwf), dim=-1)
            rays_llff = torch.reshape(rays_llff, [-1, 6])
            v.plot_test_rays(rays_nerf=rays_od,
                             rays_llff=rays_llff,
                             rays_colmap=rays_colmap,
                             mesh=mesh,
                             hwf=dataset.hwf,
                             rot_matrix=dataset.poses[i])

        hit = cast_rays_open3d(mesh, rays_od, dataset.hwf)
        
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(2,1,1)    
        ax1.imshow(dataset.images[i])
        ax2 = fig.add_subplot(2,2,1)   
        ax2.imshow(hit.T)
        plt.show()