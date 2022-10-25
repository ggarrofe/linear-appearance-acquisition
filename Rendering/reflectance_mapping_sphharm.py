import torch
import torch.nn as nn

import configargparse
import numpy as np
import visualization as v
from tqdm import tqdm
import gc
import time
import lpips
import json

import nvidia_smi
nvidia_smi.nvmlInit()

from PIL import Image

import os
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
    parser.add_argument('--encoding_freqs', type=int, help='number of frequencies used in the positional encoding', default=6)
    parser.add_argument('--deg_view', type=int, help='number of degrees used in the spherical harmonics encoding', default=3)
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
    parser.add_argument('--knn_clusters', type=int,
                        default=3, help="number of nearest clusters' colours to consider")

    args = parser.parse_args()
    return args


def compute_inv(xh, target, cluster_id, cluster_ids, embed_fn, device=torch.device("cuda"), batch_size=1e07):
    mask = cluster_ids == cluster_id
    xh, target = xh[mask], target[mask]
    del mask

    if xh.shape[0] > batch_size:
        xh, indices = utils.filter_duplicates(xh)
        target = target[indices]
        del indices

    if xh.shape[0] <= batch_size:
        xh_enc_inv = torch.linalg.pinv(embed_fn(xh.to(device)))

        linear_mapping = xh_enc_inv @ target.to(device)
    else: 
        xh_enc_inv = torch.linalg.pinv(embed_fn(xh))
        linear_mapping = xh_enc_inv @ target

    linear_mapping = linear_mapping.T.to(device)

    xh.cpu()
    target.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    return linear_mapping

def print_memory_usage():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"Device {0}: {nvidia_smi.nvmlDeviceGetName(handle)}, Memory : ({100*info.free/info.total:.2f}% free): {info.total} (total), {info.free} (free), {info.used} (used)")

def print_devices(obj):
    print(f"Devices of {type(obj).__name__}")
    for key, value in vars(obj).items():
        if torch.is_tensor(value):
            print(f"Device of {key}: {value.device}")

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
    dataset = data.NeRFDataset(args)
    dataset.compute_depths(torch.device("cpu"))
    dataset.compute_normals()
    dataset.compute_halfangles()

    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    
    # TRAINING
    embed_fn, input_ch_posenc, input_ch_sphharm = emb.get_mixed_embedder(in_dim_posenc=3, 
                                                                         in_dim_sphharm=3, 
                                                                         num_freqs=args.encoding_freqs, 
                                                                         deg_view=args.deg_view)
    input_ch=input_ch_posenc+input_ch_sphharm

    if not args.only_eval:
        x_H, target_rgb = dataset.get_X_H_rgb("train", img=-1, device="cpu")

        linear_mappings = torch.zeros([args.num_clusters, 3, input_ch]).to(device)
        cluster_ids = torch.zeros((x_H.shape[0],), dtype=torch.long).to("cpu")
        centroids = torch.zeros((args.num_clusters, 3)).to("cpu")

        mask = (x_H[:,0] == -1.) & (x_H[:,1] == -1.) & (x_H[:,2] == -1.) #not masking takes too much time
        start_time = time.time()
        cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(x_H[~mask, :3],
                                                                    num_clusters=args.num_clusters-1, 
                                                                    tol=args.kmeans_tol,
                                                                    device=device,
                                                                    batch_size=args.kmeans_batch_size,
                                                                    iter_limit=13)
        
        cluster_ids.masked_fill_(mask.to(cluster_ids.device), args.num_clusters-1)
        centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(x_H)
        lin_map_time = time.time() - start_time

        for cluster_id in tqdm(range(args.num_clusters), unit="linear mapping", desc="Computing linear mappings"):
            linear_mappings[cluster_id] = compute_inv(x_H, 
                                                    target_rgb, 
                                                    cluster_id, 
                                                    cluster_ids, 
                                                    embed_fn=embed_fn,
                                                    device=device)
        train_time = time.time() - start_time
        kmeans_time = train_time - lin_map_time
        print("Training time: %s seconds. Including %s of K-means training." % (train_time, kmeans_time))
        
        del x_H
        del target_rgb
        del cluster_ids
        torch.cuda.empty_cache()
        gc.collect()

        torch.save({ # Save our checkpoint loc
            'num_clusters': args.num_clusters,
            'linear_mappings': linear_mappings,
            'centroids': centroids,
            }, f"{args.checkpoint_path}/{args.num_clusters}clusters.tar")
    else:
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.num_clusters}clusters.tar")
        linear_mappings = checkpoint['linear_mappings']
        centroids = checkpoint['centroids']

    linear_net = net.ClusterisedLinearNetwork(linear_mappings=linear_mappings, 
                                              embed_fn=embed_fn, 
                                              num_freqs=args.encoding_freqs, 
                                              batch_size=args.batch_size,
                                              knn_clusters=args.knn_clusters,
                                              pos_boundaries=(0, input_ch_posenc),
                                              diff_boundaries=(0, input_ch_posenc),
                                              spec_boundaries=(input_ch_posenc, input_ch_posenc+input_ch_sphharm))
    linear_net.to(device)

    
    # EVALUATION
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
    print("evaluating...")
    x_H, img_tr = dataset.get_X_H_rgb("train", img=0, device=device)
    
    cluster_ids, weights = kmeans_predict(x_H[..., :3], centroids, device=device, k=args.knn_clusters)
    pred_rgb_tr = linear_net(x_H, cluster_ids, weights)
    pred_rgb_spec = linear_net.specular(x_H, cluster_ids)
    pred_rgb_diff = linear_net.diffuse(x_H, cluster_ids)
    pred_rgb_lin = linear_net.linear(x_H, cluster_ids)
        
    loss_tr = loss_fn(pred_rgb_tr, img_tr)
    print("loss train", loss_tr)

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
                        
    x_H, img_val = dataset.get_X_H_rgb("val", img=0, device=device)

    start_time = time.time()
    cluster_ids, weights = kmeans_predict(x_H[..., :3], centroids, device=device, k=args.knn_clusters)
    pred_rgb_val = linear_net(x_H, cluster_ids, weights)
    pred_time = time.time() - start_time
    print("Prediction time: %s seconds" % (pred_time))

    pred_rgb_spec = linear_net.specular(x_H, cluster_ids)
    pred_rgb_diff = linear_net.diffuse(x_H, cluster_ids)
    pred_rgb_lin = linear_net.linear(x_H, cluster_ids)

    v.validation_view_reflectance(reflectance=pred_rgb_val.detach().cpu(),
                                    specular=pred_rgb_spec.detach().cpu(),
                                    diffuse=pred_rgb_diff.detach().cpu(),
                                    linear=pred_rgb_lin.detach().cpu(),
                                    target=img_val.detach().cpu(),
                                    it=args.num_clusters,
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3),
                                    out_path=args.out_path,
                                    name=f"val_reflectance",
                                    wandb_act=False)
    
    loss_val = loss_fn(pred_rgb_val, img_val)
    print("loss val.", loss_val)

    img_shape=(dataset.hwf[0], dataset.hwf[1], 3)
    if not args.only_eval:
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
            x_H, img_val = dataset.get_X_H_rgb("val", img=i, device=device)
            start_time = time.time()
            cluster_ids, weights = kmeans_predict(x_H[..., :3], centroids, device=device, k=args.knn_clusters)
            pred_rgb_val = linear_net(x_H, cluster_ids, weights)
            
            if not os.path.exists(f"{args.out_path}/val/{args.num_clusters}clusters/"):
                os.mkdir(f"{args.out_path}/val/{args.num_clusters}clusters/")

            im_array = torch.clamp(pred_rgb_val, min=0., max=1.).detach().cpu().reshape(img_shape)
            im = Image.fromarray((im_array.numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/val/{args.num_clusters}clusters/{i}.png")


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

    img_shape=(dataset.hwf[0], dataset.hwf[1], 3)
    if "relighting" in dataset.subdirs:
        psnr_mean = 0.0
        ssim_mean = 0.0
        lpips_mean = 0.0
        pred_time_mean = 0.0

        for i in range(dataset.get_n_images("relighting")):
            x_H, img_val = dataset.get_X_H_rgb("relighting", img=i, device=device)
            start_time = time.time()
            cluster_ids_val, weights = kmeans_predict(x_H[..., :3], centroids, device=device, k=args.knn_clusters)
            pred_rgb_val = linear_net(x_H, cluster_ids_val, weights)
            pred_time = time.time() - start_time
            im_array = torch.clamp(pred_rgb_val, min=0., max=1.).detach().cpu().reshape(img_shape)
            im = Image.fromarray((im_array.numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/relighting_{i}.png")
            
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
        with open(f"{args.out_path}/relighting_results_{args.num_clusters}clusters.json", "w") as json_file:
            json.dump(results, json_file, indent = 4)

    elif dataset.get_n_images("test") > 0:

        for i in range(dataset.get_n_images("test")):
            X_H, img_test = dataset.get_X_H_rgb("test", img=i, device=device)
            cluster_ids_test, weights = kmeans_predict(X_H[..., :3], centroids, device=device, k=args.knn_clusters)
            pred_rgb_test = linear_net(X_H, cluster_ids_test, weights)
            pred_rgb_spec = linear_net.specular(X_H, cluster_ids_test)
            pred_rgb_diff = linear_net.diffuse(X_H, cluster_ids_test)

            pred_rgb_test = torch.reshape(torch.clamp(pred_rgb_test, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_test.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/reflectance/pred_{i}.png")
            pred_rgb_spec = torch.reshape(torch.clamp(pred_rgb_spec, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_spec.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/specular/pred_{i}.png")
            pred_rgb_diff = torch.reshape(torch.clamp(pred_rgb_diff, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_diff.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/diffuse/pred_{i}.png")