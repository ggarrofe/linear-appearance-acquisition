import torch
import torch.nn as nn
import numpy as np

import configargparse
import open3d as o3d
import visualization as v
from tqdm import tqdm
import gc

import matplotlib.pyplot as plt
import wandb
import sys
sys.path.append('../')


def parse_args():
    parser = configargparse.ArgumentParser(
        description="Initializes the geometry with a given mesh")
    parser.add_argument('-c', '--config', is_config_file=True,
                        help='config file path')
    parser.add_argument('--mesh_path', type=str, help='initial mesh path')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--out_path', type=str,
                        help='path to the output folder', default="./out")

    parser.add_argument('--dataset_type', type=str, help='type of dataset',
                        choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--encoding_freqs', type=int,
                        help='number of frequencies used in the positional encoding', default=6)
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    parser.add_argument('--load_light', action='store_true',
                        help='load light sources positions')
    parser.add_argument("--test", action='store_true', help='use reduced number of images')
    parser.add_argument("--only_eval", action='store_true', help='load a model and evaluate it')

    parser.add_argument('--batch_size', type=int, default=200_000, help='number of points whose rays would be used at once')
    parser.add_argument('--num_iters', type=int, help='number of iterations to train the network', default=1000)
    parser.add_argument('--num_epochs', type=int, help='number of epochs used to learn the latent features', default=100)
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument('--shuffle', type=bool, default=False)

    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument('--colab', action="store_true")

    parser.add_argument("--run_id", type=str, help='Id of the run that must be resumed')
    parser.add_argument('--checkpoint_path', type=str, help='Path where checkpoints are saved')
    parser.add_argument("--resume", action='store_true', help='Resume the run from the last checkpoint')

    parser.add_argument('--val_images', type=int, help='number of validation images', default=100)
    parser.add_argument('--train_images', type=int, help='number of training images', default=100)
    parser.add_argument('--test_images', type=int, help='number of test images', default=100)

    parser.add_argument('--num_clusters', type=int, help='number of clusters of the surface points', default=10)
    parser.add_argument('--kmeans_tol', type=float, help='threshold to stop iterating the kmeans algorithm', default=1e-04)
    parser.add_argument('--kmeans_batch_size', type=int, default=200000, help='number of points to cluster at once')

    args = parser.parse_args()
    return args


def compute_inv(xh, target, cluster_id, cluster_ids, embed_fn, device=torch.device("cuda"), batch_size=1e07):
    mask = cluster_ids == cluster_id
    xh, target = xh[mask], target[mask]
    del mask
    gc.collect()

    if xh.shape[0] < batch_size:
        xh_enc_inv = torch.linalg.pinv(embed_fn(xh.to(device)))

        linear_mapping = xh_enc_inv @ target.to(device)
    else: 
        xh, indices = utils.filter_duplicates(xh)
        target = target[indices]
        xh_enc_inv = torch.linalg.pinv(embed_fn(xh))
        linear_mapping = xh_enc_inv @ target
    return linear_mapping.T.to(device)

def predict(xh, pred, spec_pred, diffuse_pred, linear_mapping, cluster_id, cluster_ids, embed_fn):
    if not torch.any(cluster_ids == cluster_id): return

    xh_enc = embed_fn(xh[cluster_ids == cluster_id])
    pred[cluster_ids == cluster_id] = xh_enc @ linear_mapping.T

    # First half of the linear mapping will predict the diffuse color (only depends on the position)
    diffuse_pred[cluster_ids == cluster_id] = xh_enc[..., :48] @ linear_mapping[..., :48].T

    # Second half of the linear mapping will predict the specular color (depends on the half-angle vector)
    linear_mapping_spec = torch.cat([linear_mapping[..., :36], linear_mapping[..., 48:]], dim=-1)
    spec_pred[cluster_ids == cluster_id] = torch.cat([xh_enc[..., :36], xh_enc[..., 48:]], dim=-1) @ linear_mapping_spec.T

def get_x2cluster(X, cluster_ids, num_clusters):
    indices = torch.stack([torch.arange(0,cluster_ids.shape[0]), cluster_ids])
    clusters = torch.sparse_coo_tensor(indices, torch.ones((cluster_ids.shape[0],)), (cluster_ids.shape[0], num_clusters)).to_dense()
    print(clusters.shape)
    x2cluster = torch.linalg.pinv(X) @ clusters  
    print(x2cluster.shape)
    return x2cluster

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
    dataset = data.NeRFDataset(args)
    dataset.compute_depths(device=torch.device("cpu"))
    dataset.compute_normals()
    dataset.compute_halfangles()
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

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
                        "num_iters": args.num_iters,
                        "batch_size": args.batch_size,
                        "kmeans_batch_size": args.kmeans_batch_size,
                        "dataset_type": args.dataset_type,
                        "dataset_path": args.dataset_path,
                        "num_clusters": args.num_clusters
                    })

    # INITIALIZING THE LINEAR LAYER
    iter=0
    if wandb.run.resumed:
        x_NdotL_NdotH, target_rgb = dataset.get_X_NdotL_NdotH_rgb("train", img=-1, device=torch.device("cpu"))
        embed_fn, input_ch = emb.get_posenc_embedder(in_dim=x_NdotL_NdotH.shape[-1], num_freqs=6)

        reflectance_net = net.ReflectanceNetwork(in_features=x_NdotL_NdotH.shape[-1], 
                                        linear_mappings=torch.zeros((args.num_clusters, 3, input_ch)), 
                                        num_freqs=6)
        reflectance_net.to(device)
        optimizer = torch.optim.Adam(reflectance_net.parameters(), lr=args.lrate)

        wandb.restore(f"{args.checkpoint_path}/{args.run_id}.tar")
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.run_id}.tar")
        reflectance_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        centroids = checkpoint['centroids']
        iter = checkpoint['iter']
        print(f"Resuming {run.id} at iteration {iter}")

    else:
        x_NdotL_NdotH, target_rgb = dataset.get_X_NdotL_NdotH_rgb("train", img=-1, device=device)
        embed_fn, input_ch = emb.get_posenc_embedder(in_dim=x_NdotL_NdotH.shape[-1], num_freqs=6)
        
        linear_mappings = torch.zeros([args.num_clusters, 3, input_ch]).to(device)
        cluster_ids = torch.zeros((x_NdotL_NdotH.shape[0],), dtype=torch.long).to(device)
        centroids = torch.zeros((args.num_clusters, 3)).to(device)

        mask = (x_NdotL_NdotH[:,0] == -1.) & (x_NdotL_NdotH[:,1] == -1.) & (x_NdotL_NdotH[:,2] == -1.) #not masking takes too much time
        cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(x_NdotL_NdotH[~mask, :3],
                                                                    num_clusters=args.num_clusters-1, 
                                                                    tol=args.kmeans_tol,
                                                                    device=device,
                                                                    batch_size=args.kmeans_batch_size)
        
        cluster_ids.masked_fill_(mask.to(device), args.num_clusters-1)
        centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(x_NdotL_NdotH)
        cluster_ids = cluster_ids.cpu()

        for cluster_id in tqdm(range(args.num_clusters), unit="linear mapping", desc="Computing linear mappings"):
            linear_mappings[cluster_id] = compute_inv(x_NdotL_NdotH, 
                                                    target_rgb, 
                                                    cluster_id, 
                                                    cluster_ids, 
                                                    embed_fn=embed_fn,
                                                    device=device)

        reflectance_net = net.EnhancedReflectanceNetwork(in_features=x_NdotL_NdotH.shape[-1], linear_mappings=linear_mappings, num_freqs=6)
        linear_mappings = linear_mappings.cpu()
        reflectance_net = reflectance_net.to(device)

        # Create optimizer
        optimizer = torch.optim.Adam(reflectance_net.parameters(), lr=args.lrate)

    pbar = tqdm(total=args.num_iters, unit="iteration")
    pbar.update(iter)
    dataset.switch_2_X_NdotL_NdotH_dataset(device=device if args.dataset_to_gpu else torch.device("cpu"))

    while iter < args.num_iters:
        # ------------------------ TRAINING -------------------------
        reflectance_net.train()
        batch_x_NdotL_NdotH_tr, target_rgb_tr = dataset.next_batch("train", device=device)

        cluster_ids_tr = kmeans_predict(batch_x_NdotL_NdotH_tr[..., :3], centroids, device=device)
        pred_rgb = reflectance_net(batch_x_NdotL_NdotH_tr)
        pred_spec = reflectance_net.specular(batch_x_NdotL_NdotH_tr)
        pred_diff = reflectance_net.diffuse(batch_x_NdotL_NdotH_tr)
        pred_linear = reflectance_net.linear(batch_x_NdotL_NdotH_tr, cluster_ids_tr)
        '''linear_pred_rgb = reflectance_net.linear_mapping.reflectance(batch_x_NdotL_NdotH_tr.to(device), cluster_ids_tr)
        linear_pred_spec = reflectance_net.linear_mapping.specular(batch_x_NdotL_NdotH_tr.to(device), cluster_ids_tr)
        linear_pred_diff = reflectance_net.linear_mapping.diffuse(batch_x_NdotL_NdotH_tr.to(device), cluster_ids_tr)
        '''
        loss = loss_fn(pred_rgb, target_rgb_tr) #+ loss_fn(pred_spec, linear_pred_spec) + loss_fn(pred_diff, linear_pred_diff)'''
        '''print("train spec loss", 0.1 * loss_fn(pred_spec, linear_pred_spec))
        print("train diff loss", 0.1 * loss_fn(pred_diff, linear_pred_diff))
        print("train loss", loss_fn(pred_rgb, target_rgb_tr))'''
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (iter / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        wandb.log({
                "tr_loss": loss,
                "tr_psnr": mse2psnr(loss),
                "tr_linear_loss": loss_fn(pred_linear, target_rgb_tr)
                }, step=iter)

        # ------------------------ VALIDATION ------------------------
        reflectance_net.eval()
        batch_x_NdotL_NdotH_val, target_rgb_val = dataset.next_batch("val", device=device)
        
        cluster_ids_val = kmeans_predict(batch_x_NdotL_NdotH_val[..., :3], centroids, device=device)
        pred_rgb_val = reflectance_net(batch_x_NdotL_NdotH_val.to(device))
        pred_spec = reflectance_net.specular(batch_x_NdotL_NdotH_val)
        pred_diff = reflectance_net.diffuse(batch_x_NdotL_NdotH_val)
        pred_linear = reflectance_net.linear(batch_x_NdotL_NdotH_val, cluster_ids_val)
        '''linear_pred_rgb = reflectance_net.linear_mapping.reflectance(batch_x_NdotL_NdotH_val.to(device), cluster_ids_val)
        linear_pred_spec = reflectance_net.linear_mapping.specular(batch_x_NdotL_NdotH_val.to(device), cluster_ids_val)
        linear_pred_diff = reflectance_net.linear_mapping.diffuse(batch_x_NdotL_NdotH_val.to(device), cluster_ids_val)
        '''

        val_loss = loss_fn(pred_rgb_val, target_rgb_val) #+ loss_fn(pred_spec, linear_pred_spec) + loss_fn(pred_diff, linear_pred_diff)

        wandb.log({
                "val_loss": val_loss,
                "val_psnr": mse2psnr(val_loss),
                "val_linear_loss": loss_fn(pred_linear, target_rgb_val)
                }, step=iter)

        #  ------------------------ EVALUATION ------------------------
        if (iter < 200 and iter%20 == 0) or iter%2000 == 0:
            x_NdotL_NdotH, img = dataset.get_X_NdotL_NdotH_rgb("train", img=0, device=device)
            cluster_ids = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
            pred_rgb = reflectance_net(x_NdotL_NdotH)
            pred_rgb_spec = reflectance_net.specular(x_NdotL_NdotH)
            pred_rgb_diff = reflectance_net.diffuse(x_NdotL_NdotH)

            v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                          specular=pred_rgb_spec.detach().cpu(), 
                                          diffuse=pred_rgb_diff.detach().cpu(), 
                                          target=img.detach().cpu(),
                                          points=x_NdotL_NdotH[..., :3].detach().cpu(),
                                          linear=reflectance_net.linear(x_NdotL_NdotH, cluster_ids).detach().cpu(),
                                          img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                          out_path=args.out_path,
                                          it=iter,
                                          name="training_reflectance")

            x_NdotL_NdotH, img = dataset.get_X_NdotL_NdotH_rgb("val", img=0, device=device)
            cluster_ids = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
            pred_rgb = reflectance_net(x_NdotL_NdotH)
            pred_rgb_spec = reflectance_net.specular(x_NdotL_NdotH)
            pred_rgb_diff = reflectance_net.diffuse(x_NdotL_NdotH)

            v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                        specular=pred_rgb_spec.detach().cpu(), 
                                        diffuse=pred_rgb_diff.detach().cpu(), 
                                        target=img.detach().cpu(),
                                        points=x_NdotL_NdotH[..., :3].detach().cpu(),
                                        linear=reflectance_net.linear(x_NdotL_NdotH, cluster_ids).detach().cpu(),
                                        img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                        out_path=args.out_path,
                                        it=iter,
                                        name="val_reflectance")

            x_NdotL_NdotH, img = dataset.get_X_NdotL_NdotH_rgb("train", img=np.random.randint(0, dataset.get_n_images("train")), device=device)
            cluster_ids = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
            pred_rgb = reflectance_net(x_NdotL_NdotH)
            pred_rgb_spec = reflectance_net.specular(x_NdotL_NdotH)
            pred_rgb_diff = reflectance_net.diffuse(x_NdotL_NdotH)

            v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                        specular=pred_rgb_spec.detach().cpu(), 
                                        diffuse=pred_rgb_diff.detach().cpu(), 
                                        target=img.detach().cpu(),
                                        points=x_NdotL_NdotH[..., :3].detach().cpu(),
                                        linear=reflectance_net.linear(x_NdotL_NdotH, cluster_ids).detach().cpu(),
                                        img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                        out_path=args.out_path,
                                        it=iter,
                                        name="train_random_reflectance")

            x_NdotL_NdotH, img = dataset.get_X_NdotL_NdotH_rgb("val", img=np.random.randint(0, dataset.get_n_images("val")), device=device)
            cluster_ids = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
            pred_rgb = reflectance_net(x_NdotL_NdotH)
            pred_rgb_spec = reflectance_net.specular(x_NdotL_NdotH)
            pred_rgb_diff = reflectance_net.diffuse(x_NdotL_NdotH)

            v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                        specular=pred_rgb_spec.detach().cpu(), 
                                        diffuse=pred_rgb_diff.detach().cpu(), 
                                        target=img.detach().cpu(),
                                        points=x_NdotL_NdotH[..., :3].detach().cpu(),
                                        linear=reflectance_net.linear(x_NdotL_NdotH, cluster_ids).detach().cpu(),
                                        img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                        out_path=args.out_path,
                                        it=iter,
                                        name="val_random_reflectance")

            torch.save({ # Save our checkpoint loc
                'iter': iter,
                'model_state_dict': reflectance_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'centroids': centroids,
                }, f"{args.checkpoint_path}/{run.id}.tar")

            wandb.save(f"{args.checkpoint_path}/{run.id}.tar") # saves checkpoint to wandb    

        iter += 1
        pbar.update(1)
    
    pbar.close()
