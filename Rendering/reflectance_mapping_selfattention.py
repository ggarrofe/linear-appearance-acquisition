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
import wandb

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
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument('--num_iters', type=int, help='number of iterations to train the network', default=1000)

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
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    
    # TRAINING
    embed_fn, input_ch_posenc, input_ch_sphharm = emb.get_mixed_embedder(in_dim_posenc=3, in_dim_sphharm=3, num_freqs=args.encoding_freqs, deg_view=3, device=device)
    input_ch=input_ch_posenc+input_ch_sphharm
    #embed_fn, input_ch = emb.get_posenc_embedder(in_dim=6, num_freqs=args.encoding_freqs)
    
    if not args.only_eval:
        x_H, target_rgb = dataset.get_X_H_rgb("train", img=-1, device=device)
        
        linear_mappings = torch.zeros([args.num_clusters, 3, input_ch]).to(device)
        cluster_ids = torch.zeros((x_H.shape[0],), dtype=torch.long).to(device)
        centroids = torch.zeros((args.num_clusters, 3)).to(device)

        mask = (x_H[:,0] == -1.) & (x_H[:,1] == -1.) & (x_H[:,2] == -1.) #not masking takes too much time
        start_time = time.time()
        cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(x_H[~mask, :3],
                                                                    num_clusters=args.num_clusters-1, 
                                                                    tol=args.kmeans_tol,
                                                                    device=device,
                                                                    batch_size=args.batch_size)
        
        cluster_ids.masked_fill_(mask.to(device), args.num_clusters-1)
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

    else:
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.num_clusters}clusters.tar")
        linear_mappings = checkpoint['linear_mappings']
        centroids = checkpoint['centroids']

    linear_net = net.ClusterisedSelfAttention(linear_mappings=linear_mappings, 
                                              embed_fn=embed_fn, 
                                              num_freqs=6, 
                                              centroids=centroids,
                                              pos_boundaries=(0, input_ch_posenc),
                                              diff_boundaries=(0, input_ch_posenc),
                                              spec_boundaries=(input_ch_posenc, input_ch_posenc+input_ch_sphharm))
    linear_net.to(device)
    optimizer = torch.optim.Adam(linear_net.parameters(), lr=args.lrate)
    
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

    it = 0
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
    pbar = tqdm(total=args.num_iters, unit="iteration")
    pbar.update(it)
    while it < args.num_iters:
        # ------------------------ TRAINING -------------------------
        linear_net.train()
        batch_X_H_tr, target_rgb_tr = dataset.next_batch("train", device=device)

        cluster_ids_tr = kmeans_predict(batch_X_H_tr[..., :3], centroids, device=device)
        pred_rgb_tr, kmeans_rgb, attention_rgb = linear_net(batch_X_H_tr, cluster_ids_tr)
        loss_tr = loss_fn(attention_rgb, target_rgb_tr)

        optimizer.zero_grad()
        loss_tr.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (it / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        kmeans_loss = loss_fn(kmeans_rgb, target_rgb_tr)
        wandb.log({
                "tr_loss": loss_tr,
                "tr_psnr": mse2psnr(loss_tr),
                "tr_kmeans_loss": kmeans_loss,
                "tr_kmeans_psnr": mse2psnr(kmeans_loss),
                "lrate": new_lrate
                }, step=it)

        linear_net.eval()
        batch_X_H_val, target_rgb_val = dataset.next_batch("val", device=device)

        cluster_ids_val = kmeans_predict(batch_X_H_val[..., :3], centroids, device=device)
        pred_rgb_val, kmeans_rgb, attention_rgb = linear_net(batch_X_H_val, cluster_ids_val)
        loss_val = loss_fn(attention_rgb, target_rgb_val)

        kmeans_loss = loss_fn(kmeans_rgb, target_rgb_val)
        wandb.log({
                "val_loss": loss_val,
                "val_psnr": mse2psnr(loss_val),
                "val_kmeans_loss": kmeans_loss,
                "val_kmeans_psnr": mse2psnr(kmeans_loss),
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
            del kmeans_rgb
            del attention_rgb
            gc.collect()

            x_H, img_tr = dataset.get_X_H_rgb("train", img=0, device=device)
            
            cluster_ids_tr = kmeans_predict(x_H[..., :3], centroids, device=device)
            _, kmeans_rgb, attention_rgb = linear_net(x_H, cluster_ids_tr)
            kmeans_rgb, attention_rgb = kmeans_rgb.detach().cpu(), attention_rgb.detach().cpu()
            _, kmeans_spec, attention_spec = linear_net.specular(x_H, cluster_ids_tr)
            kmeans_spec, attention_spec = kmeans_spec.detach().cpu(), attention_spec.detach().cpu()
            _, kmeans_diff, attention_diff = linear_net.diffuse(x_H, cluster_ids_tr)
            kmeans_diff, attention_diff = kmeans_diff.detach().cpu(), attention_diff.detach().cpu()
                
            v.validation_view_selfattention(diffuse_att=attention_diff,
                                            specular_att=attention_spec,
                                            pred_att=attention_rgb,
                                            diffuse_kmeans=kmeans_diff,
                                            specular_kmeans=kmeans_spec,
                                            pred_kmeans=kmeans_rgb,
                                            target=img_tr,
                                            img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                            out_path=args.out_path,
                                            it=it,
                                            name="training_reflectance")
                                        

            x_H, img_val = dataset.get_X_H_rgb("val", img=0, device=device)
            start_time = time.time()
            cluster_ids_val = kmeans_predict(x_H[..., :3], centroids, device=device)
            _, kmeans_rgb, attention_rgb = linear_net(x_H, cluster_ids_val)
            pred_time = time.time() - start_time
            kmeans_rgb, attention_rgb = kmeans_rgb.detach().cpu(), attention_rgb.detach().cpu()
            _, kmeans_spec, attention_spec = linear_net.specular(x_H, cluster_ids_val)
            kmeans_spec, attention_spec = kmeans_spec.detach().cpu(), attention_spec.detach().cpu()
            _, kmeans_diff, attention_diff = linear_net.diffuse(x_H, cluster_ids_val)
            kmeans_diff, attention_diff = kmeans_diff.detach().cpu(), attention_diff.detach().cpu()
            print("Prediction time: %s seconds" % (pred_time))

            v.validation_view_selfattention(diffuse_att=attention_diff.detach().cpu(),
                                            specular_att=attention_spec.detach().cpu(),
                                            pred_att=attention_rgb.detach().cpu(),
                                            diffuse_kmeans=kmeans_diff.detach().cpu(),
                                            specular_kmeans=kmeans_spec.detach().cpu(),
                                            pred_kmeans=kmeans_rgb.detach().cpu(),
                                            target=img_val,
                                            img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                            out_path=args.out_path,
                                            it=it,
                                            name="val_reflectance")

        it += 1
        pbar.update(1)
    
    pbar.close()

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

    else:
        psnr_mean = 0.0
        ssim_mean = 0.0
        lpips_mean = 0.0
        pred_time_mean = 0.0

        for i in range(dataset.get_n_images("val")):
            x_H, img_val = dataset.get_X_H_rgb("val", img=i, device=device)
            start_time = time.time()
            cluster_ids_val = kmeans_predict(x_H[..., :3], centroids, device=device)
            pred_rgb_val = linear_net(x_H)
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