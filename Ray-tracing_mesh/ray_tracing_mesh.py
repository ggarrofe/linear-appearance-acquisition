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
    parser.add_argument('--mesh_path', type=str, help='initial mesh path')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument("--test", action='store_true', help='display tests')
    parser.add_argument('--dataset_type', type=str, help='type of dataset', choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16384)
    parser.add_argument('--colab', action="store_true")
    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument('--out_path', type=str, help='path to the output folder', default="./out")
    parser.add_argument('--shuffle', action='store_true', help='shuffle samples in dataset')
    parser.add_argument('--val_images', type=int, help='number of validation images', default=100)
    parser.add_argument('--train_images', type=int, help='number of training images', default=100)
    parser.add_argument('--test_images', type=int, help='number of test images', default=100)
    parser.add_argument('--load_light', action='store_true', help='load light sources positions')
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # python3 ray_tracing_mesh.py -c ./configs/lego.conf
    args = parse_args()
    
    if args.colab:
        sys.path.append(args.colab_path)
    import utils.data as data
    import utils.utils as utils

    device = "cpu" 
    print(f'Using {device}')

    mesh = load_mesh(args.mesh_path)
    dataset = data.NeRFDataset(args)

    for i in range(dataset.get_n_images()):
        rays_od = dataset.get_rays_od("train", i, device=device)

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        dataset.compute_depths(device=device)
        dataset.compute_normals()
        
        X, rgb, _ = dataset.get_X_target("train", i, device=device)
        #light_rays = dataset.get_light_rays("train", i, device=device)

        if args.test:
            print(dataset.poses.shape)
            v.plot_rays_and_mesh(rays_od=rays_od[320400:320401],  # 800*400+400
                                 mesh=mesh,
                                 pose=dataset.poses[i],
                                 xh=X[320400:320401])
            
            #v.render_mesh(args.mesh_path, dataset.poses[i])
            
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