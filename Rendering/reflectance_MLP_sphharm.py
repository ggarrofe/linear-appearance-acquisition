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
import wandb

import nvidia_smi
nvidia_smi.nvmlInit()

from PIL import Image

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
    parser.add_argument('--encoding_freqs', type=int, help='number of frequencies used in the positional encoding', default=6)
    parser.add_argument('--deg_view', type=int, help='number of degrees used in the spherical harmonics encoding', default=3)
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    parser.add_argument('--load_light', action='store_true',
                        help='load light sources positions')
    parser.add_argument("--test", action='store_true', help='use reduced number of images')
    parser.add_argument("--only_eval", action='store_true', help='load a model and evaluate it')

    parser.add_argument('--batch_size', type=int, default=200_000,
                        help='number of points whose rays would be used at once')
    parser.add_argument('--num_iters', type=int, help='number of iterations to train the network', default=1000)
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
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

def evaluate(dataset, args, step, lpips_vgg, device):
    mlp.eval()
    x_H, img = dataset.get_X_H_rgb("train", img=0, device=device)
    
    cluster_ids = kmeans_predict(x_H[..., :3], centroids, device=device)
    
    pred_rgb = torch.zeros((0, 3)).to(x_H)
    for i in range(0, len(x_H), 500_000):
        pred_rgb = torch.cat((pred_rgb, mlp(x_H[i:i+500_000], cluster_ids[i:i+500_000])))
    #pred_rgb_spec = linear_net.specular(x_H, cluster_ids_tr)
    #pred_rgb_diff = linear_net.diffuse(x_H, cluster_ids_tr)
    #pred_rgb_lin = linear_net.linear(x_H, cluster_ids_tr)

    img_shape = (dataset.hwf[0], dataset.hwf[1], 3)
    v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                    specular=pred_rgb.detach().cpu(), 
                                    diffuse=pred_rgb.detach().cpu(), 
                                    linear=pred_rgb.detach().cpu(),
                                    target=img.detach().cpu(),
                                    img_shape=img_shape, 
                                    out_path=args.out_path,
                                    it=step,
                                    name="reflectance_training")

    wandb.log({
            "ssim_tr": utils.compute_ssim(torch.reshape(img, img_shape),
                                        torch.reshape(pred_rgb, img_shape)),
            "lpips_tr": utils.compute_lpips(torch.reshape(img, img_shape),
                                            torch.reshape(pred_rgb, img_shape),
                                            lpips_vgg,
                                            device)
            }, step=step)

    del pred_rgb
    del x_H
    del cluster_ids
    del img
    torch.cuda.empty_cache()
    gc.collect()
    
    x_H, img = dataset.get_X_H_rgb("val", img=0, device=device)
    
    cluster_ids = kmeans_predict(x_H[..., :3], centroids, device=device)

    pred_rgb = torch.zeros((0, 3)).to(x_H)
    for i in range(0, len(x_H), 500_000):
        pred_rgb = torch.cat((pred_rgb, mlp(x_H[i:i+500_000], cluster_ids[i:i+500_000])))
    #pred_rgb_spec = linear_net.specular(x_H, cluster_ids_val)
    #pred_rgb_diff = linear_net.diffuse(x_H, cluster_ids_val)
    #pred_rgb_lin = linear_net.linear(x_H, cluster_ids_val)

    img_shape = (dataset.hwf[0], dataset.hwf[1], 3)
    v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                    specular=pred_rgb.detach().cpu(), 
                                    diffuse=pred_rgb.detach().cpu(), 
                                    linear=pred_rgb.detach().cpu(),
                                    target=img.detach().cpu(),
                                    img_shape=img_shape, 
                                    out_path=args.out_path,
                                    it=step,
                                    name="reflectance_val")

    wandb.log({
            "ssim_val": utils.compute_ssim(torch.reshape(img, img_shape),
                                        torch.reshape(pred_rgb, img_shape)),
            "lpips_val": utils.compute_lpips(torch.reshape(img, img_shape),
                                            torch.reshape(pred_rgb, img_shape),
                                            lpips_vgg,
                                            device)
            }, step=step)

    del pred_rgb
    del x_H
    del cluster_ids
    del img
    torch.cuda.empty_cache()
    gc.collect()

    x_H, img = dataset.get_X_H_rgb("train", img=np.random.randint(0, dataset.get_n_images("train")), device=device)
    
    cluster_ids = kmeans_predict(x_H[..., :3], centroids, device=device)

    pred_rgb = torch.zeros((0, 3)).to(x_H)
    for i in range(0, len(x_H), 500_000):
        pred_rgb = torch.cat((pred_rgb, mlp(x_H[i:i+500_000], cluster_ids[i:i+500_000])))
    #pred_rgb_spec = linear_net.specular(x_H, cluster_ids_tr)
    #pred_rgb_diff = linear_net.diffuse(x_H, cluster_ids_tr)
    #pred_rgb_lin = linear_net.linear(x_H, cluster_ids_tr)

    img_shape = (dataset.hwf[0], dataset.hwf[1], 3)
    v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                    specular=pred_rgb.detach().cpu(), 
                                    diffuse=pred_rgb.detach().cpu(), 
                                    linear=pred_rgb.detach().cpu(),
                                    target=img.detach().cpu(),
                                    img_shape=img_shape, 
                                    out_path=args.out_path,
                                    it=step,
                                    name="reflectance_training_random")

    del pred_rgb
    del x_H
    del cluster_ids
    del img
    torch.cuda.empty_cache()
    gc.collect()

    x_H, img = dataset.get_X_H_rgb("val", img=np.random.randint(0, dataset.get_n_images("val")), device=device)
    
    cluster_ids = kmeans_predict(x_H[..., :3], centroids, device=device)

    pred_rgb = torch.zeros((0, 3)).to(x_H)
    for i in range(0, len(x_H), 500_000):
        pred_rgb = torch.cat((pred_rgb, mlp(x_H[i:i+500_000], cluster_ids[i:i+500_000])))
    #pred_rgb_spec = linear_net.specular(x_H, cluster_ids_val)
    #pred_rgb_diff = linear_net.diffuse(x_H, cluster_ids_val)
    #pred_rgb_lin = linear_net.linear(x_H, cluster_ids_val)

    img_shape = (dataset.hwf[0], dataset.hwf[1], 3)
    v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                    specular=pred_rgb.detach().cpu(), 
                                    diffuse=pred_rgb.detach().cpu(), 
                                    linear=pred_rgb.detach().cpu(),
                                    target=img.detach().cpu(),
                                    img_shape=img_shape, 
                                    out_path=args.out_path,
                                    it=step,
                                    name="reflectance_val_random")

    del pred_rgb
    del x_H
    del cluster_ids
    del img
    torch.cuda.empty_cache()
    gc.collect()

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
    dataset.switch_2_X_H_dataset()

    print_memory_usage()

    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    
    embed_fn, input_ch_posenc, input_ch_sphharm = emb.get_mixed_embedder(in_dim_posenc=3, 
                                                                         in_dim_sphharm=3, 
                                                                         num_freqs=args.encoding_freqs, 
                                                                         deg_view=args.deg_view)
    input_ch=input_ch_posenc+input_ch_sphharm

    if not args.only_eval:
        # TRAINING
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

        x_H, target_rgb = dataset.get_X_H_rgb("train", img=-1, device="cpu")
        print_memory_usage()

        cluster_ids = torch.zeros((x_H.shape[0],), dtype=torch.long).to("cpu")
        centroids = torch.zeros((args.num_clusters, 3)).to("cpu")

        mask = (x_H[:,0] == -1.) & (x_H[:,1] == -1.) & (x_H[:,2] == -1.) #not masking takes too much time
        start_time = time.time()
        print_memory_usage()
        cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(x_H[~mask, :3],
                                                                    num_clusters=args.num_clusters-1, 
                                                                    tol=args.kmeans_tol,
                                                                    device=device,
                                                                    batch_size=args.kmeans_batch_size)
        
        print_memory_usage()
        
        centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(x_H)
        
        del x_H
        del target_rgb
        del cluster_ids
        torch.cuda.empty_cache()
        gc.collect()

        print_memory_usage()

        mlp = net.ClusterisedMLP(embed_fn=embed_fn, 
                                        input_ch=input_ch,
                                        num_clusters=args.num_clusters,
                                        num_freqs=args.encoding_freqs, 
                                        pos_boundaries=(0, input_ch_posenc),
                                        diff_boundaries=(0, input_ch_posenc),
                                        spec_boundaries=(input_ch_posenc, input_ch_posenc+input_ch_sphharm))
        mlp.to(device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lrate)
    
        it = 0
        lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
        pbar = tqdm(total=args.num_iters, unit="iteration")
        pbar.update(it)
        while it < args.num_iters:
            mlp.train()
            batch_X_H_tr, target_rgb_tr = dataset.next_batch("train", device=device)

            cluster_ids_tr = kmeans_predict(batch_X_H_tr[..., :3], centroids, device=device)
            pred_rgb_tr = mlp(batch_X_H_tr, cluster_ids_tr)
            loss_tr = loss_fn(pred_rgb_tr, target_rgb_tr)

            optimizer.zero_grad()
            loss_tr.backward()
            optimizer.step()

            decay_rate = 0.1
            decay_steps = args.lrate_decay * 1000
            new_lrate = args.lrate * (decay_rate ** (it / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            wandb.log({
                    "tr_loss": loss_tr,
                    "tr_psnr": mse2psnr(loss_tr),
                    "lrate": new_lrate
                    }, step=it)

            mlp.eval()
            batch_X_H_val, target_rgb_val = dataset.next_batch("val", device=device)

            cluster_ids_val = kmeans_predict(batch_X_H_val[..., :3], centroids, device=device)
            pred_rgb_val = mlp(batch_X_H_val, cluster_ids_val)
            loss_val = loss_fn(pred_rgb_val, target_rgb_val)

            wandb.log({
                    "val_loss": loss_val,
                    "val_psnr": mse2psnr(loss_val),
                    }, step=it)

            
            # EVALUATION
            if it%200 == 0:
                del batch_X_H_val
                del batch_X_H_tr
                del target_rgb_tr
                del target_rgb_val
                del cluster_ids_tr
                del cluster_ids_val
                del pred_rgb_val
                del pred_rgb_tr
                torch.cuda.empty_cache()
                gc.collect()

                evaluate(dataset, args, it, lpips_vgg, device)
                                        
                torch.save({ # Save our checkpoint loc
                    'num_clusters': args.num_clusters,
                    'network_state_dict': mlp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'centroids': centroids,
                    }, f"{args.checkpoint_path}/{args.num_clusters}clusters.tar")

            it += 1
            pbar.update(1)
        
        pbar.close()

    else:
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.num_clusters}clusters.tar")
        linear_mappings = checkpoint['linear_mappings']
        centroids = checkpoint['centroids']

    if dataset.get_n_images("val") > 2:
        psnr_mean = 0.0
        ssim_mean = 0.0
        lpips_mean = 0.0
        pred_time_mean = 0.0

        for i in range(dataset.get_n_images("val")):
            x_H, img_val = dataset.get_X_H_rgb("val", img=i, device=device)
            start_time = time.time()
            cluster_ids_val = kmeans_predict(x_H[..., :3], centroids, device=device)
            pred_rgb_val = mlp(x_H, cluster_ids_val)
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

    print("relighting?", dataset.subdirs)
    img_shape=(dataset.hwf[0], dataset.hwf[1], 3)
    if "relighting" in dataset.subdirs:
        print("yes")
        psnr_mean = 0.0
        ssim_mean = 0.0
        lpips_mean = 0.0
        pred_time_mean = 0.0

        for i in range(dataset.get_n_images("relighting")):
            x_H, img_val = dataset.get_X_H_rgb("relighting", img=i, device=device)
            start_time = time.time()
            cluster_ids_val = kmeans_predict(x_H[..., :3], centroids, device=device)
            pred_rgb_val = mlp(x_H, cluster_ids_val)
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
            cluster_ids_test = kmeans_predict(X_H[..., :3], centroids, device=device)
            pred_rgb_test = mlp(X_H, cluster_ids_test)
            pred_rgb_spec = mlp.specular(X_H, cluster_ids_test)
            pred_rgb_diff = mlp.diffuse(X_H, cluster_ids_test)

            pred_rgb_test = torch.reshape(torch.clamp(pred_rgb_test, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_test.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/reflectance/pred_{i}.png")
            pred_rgb_spec = torch.reshape(torch.clamp(pred_rgb_spec, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_spec.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/specular/pred_{i}.png")
            pred_rgb_diff = torch.reshape(torch.clamp(pred_rgb_diff, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_diff.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/diffuse/pred_{i}.png")