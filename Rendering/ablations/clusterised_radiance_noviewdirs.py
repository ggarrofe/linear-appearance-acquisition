import sys
sys.path.append('../')

import torch
import torch.nn as nn

import configargparse
from tqdm import tqdm
import gc
import lpips

import time
import json

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


def predict(xv, pred, linear_mapping, cluster_id, cluster_ids, embed_fn):
    mask = (cluster_ids == cluster_id)
    if not torch.any(mask):
        return

    xv_enc = embed_fn(xv[mask])
    pred[mask] = xv_enc @ linear_mapping


if __name__ == "__main__":
    args = parse_args()

    if args.colab:
        sys.path.append(args.colab_path)
        sys.path.append(args.colab_path+'Rendering/')
        print("path", sys.path)
    import utils.data as data
    import utils.networks as net
    import utils.utils as utils
    import utils.embedder as emb
    from utils.kmeans import kmeans, kmeans_predict
    import visualization as v

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
    print("only eval?", args.only_eval)
    # Load data
    print("dataset to: ", device if args.dataset_to_gpu ==
          True else torch.device("cpu"))
    dataset = data.NeRFDataset(args)
    dataset.switch_2_xn_dataset(device if args.dataset_to_gpu else torch.device("cpu"))

    loss_fn = nn.MSELoss()

    def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

    # TRAINING
    embed_fn, input_ch = emb.get_posenc_embedder(in_dim=6, num_freqs=6)
    if not args.only_eval:
        xn, target_rgb, depths = dataset.get_tensors("train", device=device)

        linear_mappings = torch.zeros([args.num_clusters, 3, input_ch]).to(device)
        cluster_ids = torch.zeros((xn.shape[0],), dtype=torch.long).to(device)
        centroids = torch.zeros((args.num_clusters, 3)).to(device)

        mask = (xn[:, 0] == -1.) & (xn[:, 1] == -1.) & (xn[:, 2] == -1.)  # not masking takes too much time

        start_time = time.time()
        cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(X=xn[~mask, :3],
                                                                    num_clusters=args.num_clusters-1,
                                                                    tol=args.kmeans_tol,
                                                                    device=device,
                                                                    batch_size=args.batch_size)

        cluster_ids.masked_fill_(mask.to(device), args.num_clusters-1)
        centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(xn)
        lin_map_time = time.time()
        for cluster_id in tqdm(range(args.num_clusters-1), unit="linear mapping", desc="Computing linear mappings"):
            linear_mappings[cluster_id] = compute_inv(xn, target_rgb, cluster_id, cluster_ids, embed_fn=embed_fn, device=device)
        
        train_time = time.time() - start_time
        kmeans_time = lin_map_time - start_time
        print("Training time: %s seconds. Including %s of K-means training." % (train_time, kmeans_time))

        xn_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)
        cluster_ids_tr = kmeans_predict(xn_tr[..., :3], centroids, device=device)
        v.plot_clusters_3Dpoints(xn_tr[..., :3], cluster_ids_tr, args.num_clusters,
                                colab=args.colab, out_path=args.out_path, filename="train_clusters.png")
    
    else:
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.num_clusters}clusters.tar")
        linear_mappings = checkpoint['linear_mappings']
        centroids = checkpoint['centroids']

    linear_net = net.ClusterisedLinearNetwork(in_features=input_ch, 
                                                linear_mappings=linear_mappings, 
                                                embed_fn=embed_fn, 
                                                num_freqs=6,
                                                input_ch=input_ch)
    linear_net.to(device)
    
    # EVALUATION
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
    xn_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)
    start_time = time.time()
    cluster_ids_tr = kmeans_predict(xn_tr[..., :3], centroids, device=device)
    pred_rgb_tr = linear_net(xn_tr, cluster_ids_tr)
    pred_time = time.time() - start_time
    print("Prediction time: %s seconds" % (pred_time))

    loss_tr = loss_fn(pred_rgb_tr, img_tr)
    img_shape = (dataset.hwf[0], dataset.hwf[1], 3)
    v.validation_view_rgb_xndv(pred_rgb_tr.detach().cpu(),
                            img_tr.detach().cpu(),
                            points=xn_tr[..., :3].detach().cpu(),
                            normals=xn_tr[..., 3:6].detach().cpu(),
                            depths=depths_tr,
                            viewdirs=torch.zeros_like(xn_tr[..., 3:6]),
                            img_shape=img_shape,
                            it=args.num_clusters,
                            out_path=args.out_path,
                            name="training_xn",
                            wandb_act=False)

    xv_val, img_val, depths_val = dataset.get_X_target("val", 0, device=device)
    cluster_ids_val = kmeans_predict(xv_val[..., :3], centroids, device=device)
    pred_rgb_val = linear_net(xv_val, cluster_ids_val)

    loss_val = loss_fn(pred_rgb_val, img_val)

    v.validation_view_rgb_xndv(pred_rgb_val.detach().cpu(),
                            img_val.detach().cpu(),
                            points=xv_val[..., :3].detach().cpu(),
                            normals=xv_val[..., 3:6].detach().cpu(),
                            depths=depths_val,
                            viewdirs=torch.zeros_like(xv_val[..., 3:6]),
                            img_shape=(dataset.hwf[0], dataset.hwf[1], 3),
                            it=args.num_clusters,
                            out_path=args.out_path,
                            name="val_xn",
                            wandb_act=False)

    if not args.only_eval:
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
            xv_val, img_val, depths_val = dataset.get_X_target("val", i, device=device)
            start_time = time.time()
            cluster_ids_val = kmeans_predict(xv_val[..., :3], centroids, device=device)
            pred_rgb_val = linear_net(xv_val, cluster_ids_val)
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