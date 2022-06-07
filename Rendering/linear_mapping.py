from ctypes import util
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import configargparse
import wandb
import time

import open3d as o3d


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
    parser.add_argument('--num_clusters', type=int, help='number of clusters of the surface points', default=10)
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    parser.add_argument('--colab', action="store_true")
    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument("--test", action='store_true', help='use reduced number of images')
    parser.add_argument("--resume", action='store_true', help='Resume the run from the last checkpoint')
    parser.add_argument("--run_id", type=str, help='Id of the run that must be resumed')
    parser.add_argument('--checkpoint_path', type=str, help='Path where checkpoints are saved')
    parser.add_argument('--val_images', type=int, help='number of validation images', default=100)
    args = parser.parse_args()
    return args

def compute_inv(xnv, target, cluster_id, cluster_ids, embed_fn):
    mask = cluster_ids == cluster_id
    xnv_enc = embed_fn(xnv[mask])
    xnv_enc_inv = torch.linalg.pinv(xnv_enc)
    rad_T = xnv_enc_inv @ target[mask]
    return rad_T

def predict(xnv, pred, mask_empty, rad_T, cluster_id, cluster_ids, embed_fn):
    mask = torch.ones((len(xnv)), dtype=torch.bool)
    mask[mask_empty] = False
    mask[~mask_empty] = (cluster_ids == cluster_id)
    if not torch.any(mask): return

    xnv_enc = embed_fn(xnv[mask])
    pred[mask] = xnv_enc @ rad_T
    

if __name__ == "__main__":
    args = parse_args()
    
    if args.colab:
        sys.path.append(args.colab_path)
    import utils.data as data
    import utils.networks as net
    import utils.utils as utils
    import utils.embedder as emb
    from utils.kmeans import kmeans, kmeans_predict
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
     
    # Load data
    print("dataset to: ", device if args.dataset_to_gpu == True else torch.device("cpu"))
    dataset = data.NeRFDataset(args, device=device if args.dataset_to_gpu else torch.device("cpu"), val_images=args.val_images)

    mesh = o3d.io.read_triangle_mesh(args.mesh, print_progress=True)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    dataset.create_xnv_dataset(scene, device=device if args.dataset_to_gpu else torch.device("cpu"))
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    embed_fn, input_ch = emb.get_embedder(in_dim=9, num_freqs=6)
    
    xnv, target_rgb, depths = dataset.get_tensors("train", device=device)
    pred_rgb = torch.zeros_like(target_rgb)

    inv_matrices = torch.zeros([args.num_clusters, input_ch, 3]).to(xnv)

    pcd_tr = []
    num_mats = 0
    
    # TRAINING
    mask = (xnv[:,0] == -1.) & (xnv[:,1] == -1.) & (xnv[:,2] == -1.)
    X = xnv[~mask, :3]

    cluster_ids = torch.zeros(X.shape[0])
    cluster_ids, centroids = kmeans(X=X,
                                    num_clusters=args.num_clusters, 
                                    tol=1e-3,
                                    distance='euclidean', 
                                    device=device,
                                    batch_size=10_000)

    for cluster_id in range(args.num_clusters):
        inv_matrices[cluster_id] = compute_inv(xnv[~mask], target_rgb[~mask], cluster_id, cluster_ids, embed_fn=embed_fn)
        
    # EVALUATION
    xnv_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)
    pred_rgb_tr = torch.zeros_like(img_tr)
    colors_tr = torch.zeros_like(xnv_tr[..., :3])

    xnv_val, img_val, depths_val = dataset.get_X_target("val", 0, device=device)
    pred_rgb_val = torch.zeros_like(img_val)

    mask_tr = (xnv_tr[:,0] == -1.) & (xnv_tr[:,1] == -1.) & (xnv_tr[:,2] == -1.)
    cluster_ids_tr = kmeans_predict(
        xnv_tr[~mask_tr, :3], centroids, 'euclidean', device=device
    )

    mask_val = (xnv_val[:,0] == -1.) & (xnv_val[:,1] == -1.) & (xnv_val[:,2] == -1.)
    cluster_ids_val = kmeans_predict(
        xnv_val[~mask_val, :3], centroids, 'euclidean', device=device
    )

    for cluster_id in range(args.num_clusters):
        predict(xnv_tr, pred_rgb_tr, mask_tr, inv_matrices[cluster_id], cluster_id, cluster_ids_tr, embed_fn)
        predict(xnv_val, pred_rgb_val, mask_val, inv_matrices[cluster_id], cluster_id, cluster_ids_val, embed_fn)
    
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

    v.validation_view_rgb_xndv(pred_rgb_tr.detach().cpu(), 
                                    img_tr.detach().cpu(), 
                                    points=xnv_tr[..., :3].detach().cpu(),
                                    normals=xnv_tr[..., 3:6].detach().cpu(),
                                    depths=depths_tr,
                                    viewdirs=xnv_tr[..., 6:].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    it=args.num_clusters, 
                                    out_path=args.out_path,
                                    name="training_xnv",
                                    wandb_act=False)

    v.validation_view_rgb_xndv(pred_rgb_val.detach().cpu(), 
                                    img_val.detach().cpu(), 
                                    points=xnv_val[..., :3].detach().cpu(),
                                    normals=xnv_val[..., 3:6].detach().cpu(),
                                    depths=depths_val,
                                    viewdirs=xnv_val[..., 6:].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    it=args.num_clusters, 
                                    out_path=args.out_path,
                                    name="val_xnv",
                                    wandb_act=False)