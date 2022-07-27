import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gc

import configargparse
import wandb

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
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    parser.add_argument('--colab', action="store_true")
    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument("--test", action='store_true', help='use reduced number of images')
    parser.add_argument("--resume", action='store_true', help='Resume the run from the last checkpoint')
    parser.add_argument("--run_id", type=str, help='Id of the run that must be resumed')
    parser.add_argument('--checkpoint_path', type=str, help='Path where checkpoints are saved')
    parser.add_argument('--val_images', type=int, help='number of validation images', default=100)
    parser.add_argument('--train_images', type=int, help='number of training images', default=100)
    parser.add_argument('--test_images', type=int, help='number of test images', default=100)
    parser.add_argument('--kmeans_tol', type=float, help='number of validation images', default=1e-04)
    parser.add_argument('--num_clusters', type=int, help='number of clusters of the surface points', default=10)
    parser.add_argument('--kmeans_batch_size', type=int, default=200000, help='number of points to cluster at once')
    parser.add_argument('--load_light', action='store_true', help='load light sources positions')
    args = parser.parse_args()

    return args

def compute_inv(xnv, target, cluster_id, cluster_ids, embed_fn, device=torch.device("cuda"), batch_size=1e07):
    mask = cluster_ids == cluster_id
    xnv, target = xnv[mask], target[mask]
    del mask
    gc.collect()

    if xnv.shape[0] < batch_size:
        xnv_enc_inv = torch.linalg.pinv(embed_fn(xnv.to(device)))
        linear_mapping = xnv_enc_inv @ target.to(device)
    else: 
        xnv, indices = utils.filter_duplicates(xnv[..., 6:])
        target = target[indices]
        xnv = xnv[indices]
        xnv_enc_inv = torch.linalg.pinv(embed_fn(xnv))
        linear_mapping = xnv_enc_inv @ target
    return linear_mapping.to(device)

def predict(xnv, pred, linear_mapping, cluster_id, cluster_ids, embed_fn):
    mask = (cluster_ids == cluster_id)
    if not torch.any(mask): return

    xnv_enc = embed_fn(xnv[mask])
    pred[mask] = xnv_enc @ linear_mapping

def diffuse_loss(points, specular, rgb, dist_th=0.03, args=None):
    diffuse_tr = rgb - specular

    points_orig = points.unsqueeze(1)
    points_dest = points.unsqueeze(0)

    dist = torch.norm(points_orig-points_dest, dim=-1)
    points_pairs = torch.argwhere(torch.triu(dist < dist_th, diagonal=1))
    loss = F.mse_loss(diffuse_tr[points_pairs[:,0]], diffuse_tr[points_pairs[:,1]])#, reduction="sum")
    '''if args is None:
        v.closepoints_pointcloud(points_pairs, points)
    else:
        v.closepoints_pointcloud(points_pairs, points, colab=args.colab, out_path=args.out_path)'''
    return loss

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
    dataset = data.NeRFDataset(args, device=device if args.dataset_to_gpu else torch.device("cpu"))
    
    mesh = o3d.io.read_triangle_mesh(args.mesh, print_progress=True)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    dataset.create_xnv_dataset(scene)

    if args.resume:
        run = wandb.init(project="controllable-neural-rendering", 
                        entity="guillemgarrofe",
                        id=args.run_id,
                        resume=True)
        
    else:
        run = wandb.init(project="controllable-neural-rendering", 
                entity="guillemgarrofe",
                config = {
                        "learning_rate": args.lrate,
                        "num_iters": args.N_iters,
                        "batch_size": args.batch_size,
                        "kmeans_batch_size": args.kmeans_batch_size,
                        "dataset_type": args.dataset_type,
                        "dataset_path": args.dataset_path,
                        "num_clusters": args.num_clusters
                    })

    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

    # Create models
    specular_net = net.SurfaceRenderingNetwork(input_ch=9,
                                           out_ch=3,
                                           hidden_ch=[512, 512, 512, 512],
                                           enc_dim=6)
    specular_net.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(specular_net.parameters(), lr=args.lrate)

    iter = 0
    if wandb.run.resumed:
        wandb.restore(f"{args.checkpoint_path}/{args.run_id}.tar")
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.run_id}.tar")
        specular_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        centroids = checkpoint['centroids']
        linear_mappings = checkpoint['linear_mappings']
        iter = checkpoint['iter']
        print(f"Resuming {run.id} at iteration {iter}")

    else:
        # ---------------------- LINEAR MAPPINGS ----------------------
        embed_fn, input_ch = emb.get_embedder(in_dim=9, num_freqs=6)
        xnv, target_rgb, depths = dataset.get_tensors("train", device=device)
        
        linear_mappings = torch.zeros([args.num_clusters, input_ch, 3]).to(device)
        cluster_ids = torch.zeros((xnv.shape[0],)).to(device)
        centroids = torch.zeros((args.num_clusters, 3)).to(device)

        mask = (xnv[:,0] == -1.) & (xnv[:,1] == -1.) & (xnv[:,2] == -1.) #not masking takes too much time
        
        cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(X=xnv[~mask, :3],
                                                                    num_clusters=args.num_clusters-1, 
                                                                    tol=args.kmeans_tol,
                                                                    device=device,
                                                                    batch_size=args.kmeans_batch_size)
        cluster_ids.masked_fill_(mask.to(device), args.num_clusters-1)
        centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(xnv)

        for cluster_id in tqdm(range(args.num_clusters-1), unit="linear mapping", desc="Computing linear mappings"):
            linear_mappings[cluster_id] = compute_inv(xnv, target_rgb, cluster_id, cluster_ids, embed_fn=embed_fn)

        #dataset.sort_clusters(cluster_ids)

    pbar = tqdm(total=args.N_iters, unit="iteration")
    pbar.update(iter)
    while iter < args.N_iters:
        # By default each batch will correspond to the rays of a single image
        batch_xnv_tr, target_rgb_tr = dataset.next_batch("train", device=device)
        
        specular_net.train()
        specular_tr = specular_net(batch_xnv_tr[..., :3], batch_xnv_tr[..., 3:6], batch_xnv_tr[..., 6:])
        
        cluster_ids_tr = kmeans_predict(batch_xnv_tr[..., :3], centroids, device=device)
        pred_rgb_tr = torch.zeros_like(target_rgb_tr)
        for cluster_id in range(args.num_clusters-1):
            predict(batch_xnv_tr, pred_rgb_tr, linear_mappings[cluster_id], cluster_id, cluster_ids_tr, embed_fn)

        loss = diffuse_loss(batch_xnv_tr[..., :3], specular_tr, pred_rgb_tr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (iter / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        wandb.log({
                "loss": loss
                }, step=iter)

        #  --------------- VALIDATION --------------

        specular_net.eval()
        batch_xnv_val, target_rgb_val = dataset.next_batch("val", device=device)
        specular_val = specular_net(batch_xnv_val[..., :3], batch_xnv_val[..., 3:6], batch_xnv_val[..., 6:])

        cluster_ids_val = kmeans_predict(batch_xnv_val[..., :3], centroids, device=device)
        pred_rgb_val = torch.zeros_like(target_rgb_val)
        for cluster_id in range(args.num_clusters-1):
            predict(batch_xnv_val, pred_rgb_val, linear_mappings[cluster_id], cluster_id, cluster_ids_val, embed_fn)

        val_loss = diffuse_loss(batch_xnv_val[..., :3], specular_val, pred_rgb_val)

        wandb.log({
                "val_loss": val_loss
                }, step=iter)

        #  --------------- EVALUATION --------------
        if iter%200 == 0:
            xnv, img, depths = dataset.get_X_target("train", 0, device=device)
            specular_map = None
            reflectance_map = None
            for i in range(0, xnv.shape[0], args.batch_size):
                batch = xnv[i:i+args.batch_size]
                specular_pred = specular_net(batch[..., :3], batch[..., 3:6], batch[..., 6:])
                cluster_ids_pr = kmeans_predict(batch[..., :3], centroids, device=device)
                
                rgb_pred = torch.zeros((batch.shape[0], 3)).to(batch)
                for cluster_id in range(args.num_clusters-1):
                    predict(batch, rgb_pred, linear_mappings[cluster_id], cluster_id, cluster_ids_pr, embed_fn)

                if specular_map is None:
                    specular_map = specular_pred
                    diffuse_map = rgb_pred - specular_pred
                    reflectance_map = rgb_pred
                else:
                    specular_map = torch.cat((specular_map, specular_pred), dim=0)
                    diffuse_map = torch.cat((diffuse_map, rgb_pred - specular_pred), dim=0)
                    reflectance_map = torch.cat((reflectance_map, rgb_pred), dim=0)

            v.validation_view_reflectance(reflectance=reflectance_map.detach().cpu(),
                                          specular=specular_map.detach().cpu(),
                                          diffuse=diffuse_map.detach().cpu(),
                                          target=img.detach().cpu(), 
                                          points=xnv[..., :3].detach().cpu(),
                                          img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                          it=iter, 
                                          out_path=args.out_path,
                                          name="training_reflectance")

            xnv, img, depths = dataset.get_X_target("train", np.random.randint(0, dataset.get_n_images()), device=device)
            specular_map = None
            reflectance_map = None
            for i in range(0, xnv.shape[0], args.batch_size):
                batch = xnv[i:i+args.batch_size]
                specular_pred = specular_net(batch[..., :3], batch[..., 3:6], batch[..., 6:])
                cluster_ids_pr = kmeans_predict(batch[..., :3], centroids, device=device)
                
                rgb_pred = torch.zeros((batch.shape[0], 3)).to(batch)
                for cluster_id in range(args.num_clusters-1):
                    predict(batch, rgb_pred, linear_mappings[cluster_id], cluster_id, cluster_ids_pr, embed_fn)
                
                if specular_map is None:
                    specular_map = specular_pred
                    diffuse_map = rgb_pred - specular_pred
                    reflectance_map = rgb_pred
                else:
                    specular_map = torch.cat((specular_map, specular_pred), dim=0)
                    diffuse_map = torch.cat((diffuse_map, rgb_pred - specular_pred), dim=0)
                    reflectance_map = torch.cat((reflectance_map, rgb_pred), dim=0)

            v.validation_view_reflectance(reflectance=reflectance_map.detach().cpu(),
                                          specular=specular_map.detach().cpu(),
                                          diffuse=diffuse_map.detach().cpu(),
                                          target=img.detach().cpu(), 
                                          points=xnv[..., :3].detach().cpu(),
                                          img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                          it=iter, 
                                          out_path=args.out_path,
                                          name="training_random_reflectance")

            xnv, img, depths = dataset.get_X_target("val", 0, device=device)
            specular_map = None
            reflectance_map = None
            for i in range(0, xnv.shape[0], args.batch_size):
                batch = xnv[i:i+args.batch_size]
                specular_pred = specular_net(batch[..., :3], batch[..., 3:6], batch[..., 6:])
                cluster_ids_pr = kmeans_predict(batch[..., :3], centroids, device=device)
                
                rgb_pred = torch.zeros((batch.shape[0], 3)).to(batch)
                for cluster_id in range(args.num_clusters-1):
                    predict(batch, rgb_pred, linear_mappings[cluster_id], cluster_id, cluster_ids_pr, embed_fn)
                
                if specular_map is None:
                    specular_map = specular_pred
                    diffuse_map = rgb_pred - specular_pred
                    reflectance_map = rgb_pred
                else:
                    specular_map = torch.cat((specular_map, specular_pred), dim=0)
                    diffuse_map = torch.cat((diffuse_map, rgb_pred - specular_pred), dim=0)
                    reflectance_map = torch.cat((reflectance_map, rgb_pred), dim=0)

            v.validation_view_reflectance(reflectance=reflectance_map.detach().cpu(),
                                          specular=specular_map.detach().cpu(),
                                          diffuse=diffuse_map.detach().cpu(),
                                          target=img.detach().cpu(), 
                                          points=xnv[..., :3].detach().cpu(),
                                          img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                          it=iter, 
                                          out_path=args.out_path,
                                          name="val_reflectance")

            xnv, img, depths = dataset.get_X_target("val", np.random.randint(0, dataset.get_n_images("val")), device=device)
            specular_map = None
            reflectance_map = None
            for i in range(0, xnv.shape[0], args.batch_size):
                batch = xnv[i:i+args.batch_size]
                specular_pred = specular_net(batch[..., :3], batch[..., 3:6], batch[..., 6:])
                cluster_ids_pr = kmeans_predict(batch[..., :3], centroids, device=device)
                
                rgb_pred = torch.zeros((batch.shape[0], 3)).to(batch)
                for cluster_id in range(args.num_clusters-1):
                    predict(batch, rgb_pred, linear_mappings[cluster_id], cluster_id, cluster_ids_pr, embed_fn)
                
                if specular_map is None:
                    specular_map = specular_pred
                    diffuse_map = rgb_pred - specular_pred
                    reflectance_map = rgb_pred
                else:
                    specular_map = torch.cat((specular_map, specular_pred), dim=0)
                    diffuse_map = torch.cat((diffuse_map, rgb_pred - specular_pred), dim=0)
                    reflectance_map = torch.cat((reflectance_map, rgb_pred), dim=0)

            v.validation_view_reflectance(reflectance=reflectance_map.detach().cpu(),
                                          specular=specular_map.detach().cpu(),
                                          diffuse=diffuse_map.detach().cpu(),
                                          target=img.detach().cpu(), 
                                          points=xnv[..., :3].detach().cpu(),
                                          img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                          it=iter, 
                                          out_path=args.out_path,
                                          name="val_random_reflectance")

            torch.save({ # Save our checkpoint loc
                'iter': iter,
                'model_state_dict': specular_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'centroids': centroids,
                'linear_mappings': linear_mappings
                }, f"{args.checkpoint_path}/{run.id}.tar")
            wandb.save(f"{args.checkpoint_path}/{run.id}.tar") # saves checkpoint to wandb    

        iter += 1
        pbar.update(1)
    
    pbar.close()