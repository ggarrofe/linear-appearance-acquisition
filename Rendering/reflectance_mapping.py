from tracemalloc import start
import torch
import torch.nn as nn

import configargparse
import open3d as o3d
import visualization as v
from tqdm import tqdm
import gc
import time
import lpips
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

print(sys.path)

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

    parser.add_argument('--batch_size', type=int, default=200_000,
                        help='number of points whose rays would be used at once')
    parser.add_argument('--shuffle', type=bool, default=False)

    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument('--colab', action="store_true")

    parser.add_argument('--checkpoint_path', type=str,
                        help='Path where checkpoints are saved')

    parser.add_argument('--val_images', type=int,
                        help='number of validation images', default=100)
    parser.add_argument('--train_images', type=int,
                        help='number of training images', default=100)
    parser.add_argument('--test_images', type=int,
                        help='number of test images', default=100)

    parser.add_argument('--num_clusters', type=int,
                        help='number of clusters of the surface points', default=10)
    parser.add_argument('--kmeans_tol', type=float,
                        help='threshold to stop iterating the kmeans algorithm', default=1e-04)
    parser.add_argument('--kmeans_batch_size', type=int,
                        default=200000, help='number of points to cluster at once')

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
    dataset.compute_depths(torch.device("cpu"))
    dataset.compute_normals()
    dataset.compute_halfangles()
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    
    # TRAINING
    embed_fn, input_ch = emb.get_posenc_embedder(in_dim=5, num_freqs=6)
    if not args.only_eval:
        x_NdotL_NdotH, target_rgb = dataset.get_X_NdotL_NdotH_rgb("train", img=-1, device=device)
        
        linear_mappings = torch.zeros([args.num_clusters, 3, input_ch]).to(device)
        cluster_ids = torch.zeros((x_NdotL_NdotH.shape[0],), dtype=torch.long).to(device)
        centroids = torch.zeros((args.num_clusters, 3)).to(device)

        mask = (x_NdotL_NdotH[:,0] == -1.) & (x_NdotL_NdotH[:,1] == -1.) & (x_NdotL_NdotH[:,2] == -1.) #not masking takes too much time
        start_time = time.time()
        cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(x_NdotL_NdotH[~mask, :3],
                                                                    num_clusters=args.num_clusters-1, 
                                                                    tol=args.kmeans_tol,
                                                                    device=device,
                                                                    batch_size=args.batch_size)
        
        cluster_ids.masked_fill_(mask.to(device), args.num_clusters-1)
        centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(x_NdotL_NdotH)
        lin_map_time = time.time() - start_time

        for cluster_id in tqdm(range(args.num_clusters), unit="linear mapping", desc="Computing linear mappings"):
            linear_mappings[cluster_id] = compute_inv(x_NdotL_NdotH, 
                                                    target_rgb, 
                                                    cluster_id, 
                                                    cluster_ids, 
                                                    embed_fn=embed_fn,
                                                    device=device)
        train_time = time.time() - start_time
        kmeans_time = train_time - lin_map_time
        print("Training time: %s seconds. Including %s of K-means training." % (train_time, kmeans_time))

    else:
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.num_clusters}clusters.tar")
        linear_mappings = checkpoint['linear_mappings']
        centroids = checkpoint['centroids']

    linear_net = net.ClusterisedLinearNetwork(linear_mappings=linear_mappings, 
                                                num_freqs=args.encoding_freqs,
                                                embed_fn=embed_fn,
                                                pos_boundaries=(0, 3*2*args.encoding_freqs),
                                                diff_boundaries=(0, 4*2*args.encoding_freqs),
                                                spec_boundaries=(4*2*args.encoding_freqs, 5*2*args.encoding_freqs))
    linear_net.to(device)
    
    # EVALUATION
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
    print("evaluating...")
    x_NdotL_NdotH, img_tr = dataset.get_X_NdotL_NdotH_rgb("train", img=0, device=device)
    
    cluster_ids_tr = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
    pred_rgb_tr = linear_net(x_NdotL_NdotH, cluster_ids_tr)
    pred_rgb_spec = linear_net.specular(x_NdotL_NdotH, cluster_ids_tr)
    pred_rgb_diff = linear_net.diffuse(x_NdotL_NdotH, cluster_ids_tr)
    pred_rgb_lin = linear_net.linear(x_NdotL_NdotH, cluster_ids_tr)
    
    spec_comp = linear_net.specular_component(x_NdotL_NdotH, cluster_ids_tr)
    diff_comp = linear_net.diffuse_component(x_NdotL_NdotH, cluster_ids_tr)
    amb_comp = linear_net.ambient_component(x_NdotL_NdotH, cluster_ids_tr)
        
    loss_tr = loss_fn(pred_rgb_tr, img_tr)

    v.validation_view_reflectance(reflectance=pred_rgb_tr.detach().cpu(),
                                specular=pred_rgb_spec.detach().cpu(), 
                                diffuse=pred_rgb_diff.detach().cpu(), 
                                linear=pred_rgb_lin.detach().cpu(),
                                target=img_tr.detach().cpu(),
                                img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                out_path=args.out_path,
                                it=args.num_clusters,
                                name="training_reflectance",
                                wandb_act=False)
                                
    for i in range(2):
        x_NdotL_NdotH, img_val = dataset.get_X_NdotL_NdotH_rgb("val", img=i, device=device)
        start_time = time.time()
        cluster_ids_tr = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
        pred_rgb_val = linear_net(x_NdotL_NdotH, cluster_ids_tr)
        pred_time = time.time() - start_time
        print("Prediction time: %s seconds" % (pred_time))

        pred_rgb_spec = linear_net.specular(x_NdotL_NdotH, cluster_ids_tr)
        pred_rgb_diff = linear_net.diffuse(x_NdotL_NdotH, cluster_ids_tr)
        pred_rgb_lin = linear_net.linear(x_NdotL_NdotH, cluster_ids_tr)

        v.validation_view_reflectance(reflectance=pred_rgb_val.detach().cpu(),
                                        specular=pred_rgb_spec.detach().cpu(),
                                        diffuse=pred_rgb_diff.detach().cpu(),
                                        linear=pred_rgb_lin.detach().cpu(),
                                        target=img_val.detach().cpu(),
                                        it=args.num_clusters,
                                        img_shape=(dataset.hwf[0], dataset.hwf[1], 3),
                                        out_path=args.out_path,
                                        name=f"val_reflectance_{i}",
                                        wandb_act=False)
    
    loss_val = loss_fn(pred_rgb_val, img_val)
    img_shape=(dataset.hwf[0], dataset.hwf[1], 3)
    print("Saving results?", args.only_eval)
    if not args.only_eval:
        print("yes")
        results = {
            "loss_tr": loss_tr.item(),
            "psnr_tr": mse2psnr(loss_tr).item(),
            "ssim_tr": utils.compute_ssim(torch.reshape(img_tr, img_shape),
                                        torch.reshape(pred_rgb_tr, img_shape)),
            "lpips_tr": utils.compute_lpips(torch.reshape(img_tr, img_shape),
                                            torch.reshape(pred_rgb_tr, img_shape),
                                            lpips_vgg,
                                            device),
            "loss_val": loss_val.item(),
            "psnr_val": mse2psnr(loss_val).item(),
            "ssim_val": utils.compute_ssim(torch.reshape(img_val, img_shape),
                                        torch.reshape(pred_rgb_val, img_shape)),
            "lpips_val": utils.compute_lpips(torch.reshape(img_val, img_shape),
                                            torch.reshape(pred_rgb_val, img_shape),
                                            lpips_vgg,
                                            device),
            "train_time": train_time,
            "kmeans_time": kmeans_time,
            "pred_time": pred_time
        }

        with open(f"{args.out_path}/results_{args.num_clusters}clusters.json", "w") as json_file:
            json.dump(results, json_file, indent = 4)

        torch.save({ # Save our checkpoint loc
            'num_clusters': args.num_clusters,
            'linear_mappings': linear_mappings,
            'centroids': centroids,
            }, f"{args.checkpoint_path}/{args.num_clusters}clusters.tar")

    elif dataset.get_n_images("val") > 2:
        psnr_mean = 0.0
        ssim_mean = 0.0
        lpips_mean = 0.0
        pred_time_mean = 0.0

        for i in range(dataset.get_n_images("val")):
            x_NdotL_NdotH, img_val = dataset.get_X_NdotL_NdotH_rgb("val", img=i, device=device)
            start_time = time.time()
            cluster_ids_val = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
            pred_rgb_val = linear_net(x_NdotL_NdotH, cluster_ids_val)
            pred_time = time.time() - start_time
            
            pred_time_mean += (pred_time - pred_time_mean)/(i+1)
            loss_val = loss_fn(pred_rgb_val, img_val)
            psnr_mean += (mse2psnr(loss_val).item() - psnr_mean)/(i+1)
            ssim = utils.compute_ssim(torch.reshape(img_val, img_shape),
                                       torch.reshape(pred_rgb_val, img_shape))
            ssim_mean += (ssim - ssim_mean)/(i+1)

            lpips_val = utils.compute_lpips(torch.reshape(img_val, img_shape),
                                         torch.reshape(pred_rgb_val, img_shape),
                                         lpips_vgg,
                                         device)
            lpips_mean += (lpips_val - lpips_mean)/(i+1)

        results = {
            "psnr_mean": psnr_mean,
            "ssim_mean": ssim_mean,
            "lpips_mean": lpips_mean,
            "pred_time_mean": pred_time_mean
        }
        with open(f"{args.out_path}/val_results_{args.num_clusters}clusters.json", "w") as json_file:
            json.dump(results, json_file, indent = 4)

    elif dataset.get_n_images("test") > 0:

        for i in range(dataset.get_n_images("test")):
            x_NdotL_NdotH, img_test = dataset.get_X_NdotL_NdotH_rgb("test", img=i, device=device)
            cluster_ids_test = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
            pred_rgb_test = linear_net(x_NdotL_NdotH, cluster_ids_test)
            pred_rgb_spec = linear_net.specular(x_NdotL_NdotH, cluster_ids_test)
            pred_rgb_diff = linear_net.diffuse(x_NdotL_NdotH, cluster_ids_test)

            pred_rgb_test = torch.reshape(torch.clamp(pred_rgb_test, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_test.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/reflectance/pred_{100+i}.png")
            pred_rgb_spec = torch.reshape(torch.clamp(pred_rgb_spec, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_spec.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/specular/pred_{100+i}.png")
            pred_rgb_diff = torch.reshape(torch.clamp(pred_rgb_diff, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_diff.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/diffuse/pred_{100+i}.png")