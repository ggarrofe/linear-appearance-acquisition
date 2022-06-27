import configargparse
import torch
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d

import visualization as v

import sys
sys.path.append('../')

def load_mesh(path):
    return o3d.io.read_triangle_mesh(path, print_progress=True)

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
    parser.add_argument('--colab', action="store_true")
    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument('--out_path', type=str, help='path to the output folder', default="./out")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # python3 ray_tracing_mesh.py -c ./configs/lego.conf
    args = parse_args()
    
    if args.colab:
        sys.path.append(args.colab_path)
    import utils.data as data
    import utils.utils as utils

    device = "cpu" #torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')

    mesh = load_mesh(args.mesh)
    dataset = data.NeRFDataset(args, device=device)

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    for i in range(dataset.get_n_images()):
        rays_od = dataset.get_rays_od("train", i, device=device)

        if args.test_path is None:
            #v.plot_rays_and_mesh(rays_od, mesh, dataset.hwf)
            pass
            
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
            '''v.plot_test_rays(rays_nerf=rays_od,
                             rays_llff=rays_llff,
                             rays_colmap=rays_colmap,
                             mesh=mesh,
                             hwf=dataset.hwf,
                             rot_matrix=dataset.poses[i])'''

        hit = utils.cast_rays(scene, rays_od)
        hit = hit.reshape(dataset.hwf[0], dataset.hwf[1])
        img = dataset.get_image("train", i, device=device).numpy()
        img = img.reshape(dataset.hwf[0], dataset.hwf[1], img.shape[-1])
        
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(2,1,1)    
        ax1.imshow(img)
        ax2 = fig.add_subplot(2,2,1)   
        ax2.imshow(hit.T)
        plt.savefig(args.out_path+f"/ray_cast_{i}.png")
        plt.show()