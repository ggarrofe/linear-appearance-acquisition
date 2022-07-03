from ctypes import util
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import configargparse
import wandb

import open3d as o3d

import nerf
import visualization as v

import matplotlib.pyplot as plt

import sys
sys.path.append('../')

def parse_args():
    parser = configargparse.ArgumentParser(description="Initializes the geometry with a given mesh")
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--mesh', type=str, help='initial mesh path')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--out_path', type=str, help='path to the output folder', default="./out")
    parser.add_argument('--dataset_type', type=str, help='type of dataset', choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16384, help='number of images whose rays would be used at once')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument('--N_iters', type=int, help='number of iterations to train the network', default=1000)
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    parser.add_argument('--colab', action="store_true")
    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument("--test", action='store_true', help='use reduced number of images')
    parser.add_argument("--resume", action='store_true', help='Resume the run from the last checkpoint')
    parser.add_argument("--run_id", type=str, help='Id of the run that must be resumed')
    parser.add_argument('--checkpoint_path', type=str, help='Path where checkpoints are saved')
    args = parser.parse_args()
    return args

def compute_inv(point, xnv, target, step, embed_fn, input_ch):
    mask = torch.norm(xnv[..., :3] - point, dim=-1) <= np.sqrt(3 * ((step/2)**2))
    if not torch.any(mask): return torch.zeros((input_ch,3))

    xnv_enc = embed_fn(xnv)
    print("encoding shape", xnv_enc[mask].shape, point)
    
    xnv_enc_inv = torch.linalg.pinv(xnv_enc[mask])
    rad_T = xnv_enc_inv @ target[mask]
    return rad_T

def predict(point, xnv, pred, rad_T, step, embed_fn, input_ch):
    mask = torch.norm(xnv[..., :3] - point, dim=-1) <= np.sqrt(3 * ((step/2)**2))
    if not torch.any(mask): return torch.zeros((input_ch,3))

    xnv_enc = embed_fn(xnv)
    pred[mask] = xnv_enc[mask] @ rad_T
    return mask
    

if __name__ == "__main__":
    args = parse_args()
    
    if args.colab:
        sys.path.append(args.colab_path)
    import utils.data as data
    import utils.networks as net
    import utils.utils as utils
    import utils.embedder as emb
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
     
    # Load data
    print("dataset to: ", device if args.dataset_to_gpu == True else torch.device("cpu"))
    dataset = data.NeRFDataset(args, device=device if args.dataset_to_gpu else torch.device("cpu"))
    
    mesh = o3d.io.read_triangle_mesh(args.mesh, print_progress=True)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    dataset.create_xnv_dataset(scene)
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    embed_fn, input_ch = emb.get_embedder(in_dim=9, num_freqs=6)
    
    vol_step = 0.5
    
    xnv, target_rgb, depths = dataset.get_tensors("train", device=device)
    pred_rgb = torch.zeros_like(target_rgb)

    range_x = np.arange(torch.min(xnv[..., 0]), torch.max(xnv[..., 0]), vol_step)
    range_y = np.arange(torch.min(xnv[..., 1]), torch.max(xnv[..., 1]), vol_step)
    range_z = np.arange(torch.min(xnv[..., 2]), torch.max(xnv[..., 2]), vol_step)
    inv_matrices = torch.zeros([len(list(range_x)), len(list(range_y)), len(list(range_z)), input_ch, 3])

    xnv_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)
    colors_tr = torch.zeros_like(xnv_tr[..., :3])
    pred_rgb_tr = torch.zeros_like(img_tr)
    xnv_val, img_val, depths_val = dataset.get_X_target("val", 0, device=device)
    pred_rgb_val = torch.zeros_like(img_val)

    pbar = tqdm(total=inv_matrices.shape[0]*inv_matrices.shape[1]*inv_matrices.shape[2], unit="iteration")
    pcd_tr = []
    for i, x in enumerate(range_x):
        for j, y in enumerate(range_y):
            for k, z in enumerate(range_z):
                inv_matrices[i,j,k] = compute_inv(point=torch.tensor([x, y, z]),
                                                  xnv=xnv,
                                                  target=target_rgb,
                                                  step=vol_step,
                                                  embed_fn=embed_fn,
                                                  input_ch=input_ch)
                predict(torch.tensor([x, y, z]), xnv, pred_rgb, inv_matrices[i,j,k], vol_step, embed_fn, input_ch)
                mask = predict(torch.tensor([x, y, z]), xnv_tr, pred_rgb_tr, inv_matrices[i,j,k], vol_step, embed_fn, input_ch)
                predict(torch.tensor([x, y, z]), xnv_val, pred_rgb_val, inv_matrices[i,j,k], vol_step, embed_fn, input_ch)
                
                # Visualization of surface clusters
                if torch.any(mask):
                    color = torch.rand(size=(3,))
                    colors_tr[mask] = color.expand(xnv_tr[mask, :3].shape)

                pbar.update(1)

    pcd_tr = o3d.t.geometry.PointCloud(utils.torch2open3d(xnv_tr[..., :3]))
    pcd_tr.point["colors"] = utils.torch2open3d(colors_tr)
    o3d.visualization.draw_geometries([pcd_tr.to_legacy()])
    
    # By default each batch will correspond to the rays of a single image
    #
    '''point=torch.tensor([0.4863, 0.9986, 1.2043])
    xnv_tr, target_rgb_tr, depths = dataset.get_tensors("train", device=device)
    pred_rgb = torch.zeros_like(target_rgb_tr)
    
    xnv_enc = embed_fn(xnv_tr)

    mask = torch.norm(xnv_tr[..., :3] - point, dim=-1) < 0.5

    print("Computing linear mapping...")
    print("encoding shape", xnv_enc[mask].shape)
    
    xnv_enc_inv = torch.linalg.pinv(xnv_enc[mask])
    rad_T = xnv_enc_inv @ target_rgb_tr[mask]

    pred_rgb[mask] = xnv_enc[mask] @ rad_T'''
    loss = loss_fn(pred_rgb, target_rgb)
    loss_tr = loss_fn(pred_rgb_tr, img_tr)
    loss_val = loss_fn(pred_rgb_val, img_val)

    print({
            "loss": loss,
            "psnr": mse2psnr(loss),
            "loss_tr": loss_tr,
            "psnr_tr": mse2psnr(loss_tr),
            "loss_val": loss_val,
            "psnr_val": mse2psnr(loss_val)
            })

    '''xnv_tr, img, depths = dataset.get_X_target("train", 0, device=device)
    pred_rgb = torch.zeros_like(img)
    mask = torch.norm(xnv_tr[..., :3] - point, dim=-1) < 0.5
    xnv_enc = embed_fn(xnv_tr)
    pred_rgb[mask] = xnv_enc[mask] @ rad_T'''

    v.validation_view_rgb_xndv(pred_rgb_tr.detach().cpu(), 
                                    img_tr.detach().cpu(), 
                                    points=xnv_tr[..., :3].detach().cpu(),
                                    normals=xnv_tr[..., 3:6].detach().cpu(),
                                    depths=depths_tr,
                                    viewdirs=xnv_tr[..., 6:].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    it=0, 
                                    out_path=args.out_path,
                                    name="training_xnv",
                                    wandb_act=False)

    '''xnv_tr, img, depths = dataset.get_X_target("val", 0, device=device)
    xnv_enc = embed_fn(xnv_tr)
    pred_rgb = xnv_enc @ rad_T'''

    v.validation_view_rgb_xndv(pred_rgb_val.detach().cpu(), 
                                    img_val.detach().cpu(), 
                                    points=xnv_val[..., :3].detach().cpu(),
                                    normals=xnv_val[..., 3:6].detach().cpu(),
                                    depths=depths_val,
                                    viewdirs=xnv_val[..., 6:].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    it=0, 
                                    out_path=args.out_path,
                                    name="val_xnv",
                                    wandb_act=False)

        