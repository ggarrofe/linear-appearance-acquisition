import torch
import torch.nn as nn
import numpy as np

import configargparse
import visualization as v
from tqdm import tqdm
import gc
import lpips

import time
import json

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
    parser.add_argument("--test", action='store_true',
                        help='use reduced number of images')

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

    args = parser.parse_args()
    return args

def compute_inv(xnv, target, embed_fn, device=torch.device("cuda"), gpu_limit=5e06, ram_limit=1e07):
    if xnv.shape[0] >= gpu_limit:
        xnv, indices = utils.filter_duplicates(xnv)
        target = target[indices]

    if xnv.shape[0] < gpu_limit:
        xnv_enc_inv = torch.linalg.pinv(embed_fn(xnv.to(device)))
        linear_mapping = xnv_enc_inv @ target.to(device)
    else:
        if xnv.shape[0] >= ram_limit:
            perm = torch.randperm(xnv.shape[0])
            idx = perm[:int(ram_limit)]
            xnv = xnv[idx]
            target = target[idx]
            del perm
            del idx
            gc.collect()

        xnv_enc_inv = torch.linalg.pinv(embed_fn(xnv.cpu()))
        linear_mapping = xnv_enc_inv @ target.cpu()
        
    return linear_mapping.T.to(device)

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
    dataset = data.NeRFDataset(args)
    dataset.switch_2_xnv_dataset(
        device if args.dataset_to_gpu else torch.device("cpu"))

    loss_fn = nn.MSELoss()

    def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

    # TRAINING
    embed_fn, input_ch = emb.get_posenc_embedder(in_dim=9, num_freqs=6)
    xnv, target_rgb, depths = dataset.get_tensors("train", device=device)
    
    linear_mapping = torch.zeros([3, input_ch]).to(device)
    start_time = time.time()

    linear_mapping = compute_inv(xnv, 
                                 target_rgb,
                                 embed_fn=embed_fn,
                                 device=device)
    print("linear mapping shape", linear_mapping.shape)
    train_time = time.time() - start_time
    print("Training time: %s seconds." % (train_time))

    xnv_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)
    linear_net = net.LinearMapping(in_features=xnv.shape[-1], 
                                   linear_mappings=linear_mapping, 
                                   num_freqs=6)
    linear_net.to(device)

    # EVALUATION
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
    img_shape = (dataset.hwf[0], dataset.hwf[1], 3)
    xnv_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)

    start_time = time.time()
    pred_rgb_tr = linear_net(embed_fn(xnv_tr))
    print("pred rgb tr", pred_rgb_tr.shape)
    pred_time = time.time() - start_time
    print("Prediction time: %s seconds" % (pred_time))

    loss_tr = loss_fn(pred_rgb_tr, img_tr)
    v.validation_view_rgb_xndv(pred_rgb_tr.detach().cpu(),
                               img_tr.detach().cpu(),
                               points=xnv_tr[..., :3].detach().cpu(),
                               normals=xnv_tr[..., 3:6].detach().cpu(),
                               depths=depths_tr,
                               viewdirs=xnv_tr[..., 6:].detach().cpu(),
                               img_shape=img_shape,
                               it=1,
                               out_path=args.out_path,
                               name="training_xnv",
                               wandb_act=False)

    xnv_val, img_val, depths_val = dataset.get_X_target("val", 0, device=device)
    pred_rgb_val = linear_net(embed_fn(xnv_val))
    loss_val = loss_fn(pred_rgb_val, img_val)

    v.validation_view_rgb_xndv(pred_rgb_val.detach().cpu(),
                               img_val.detach().cpu(),
                               points=xnv_val[..., :3].detach().cpu(),
                               normals=xnv_val[..., 3:6].detach().cpu(),
                               depths=depths_val,
                               viewdirs=xnv_val[..., 6:].detach().cpu(),
                               img_shape=(dataset.hwf[0], dataset.hwf[1], 3),
                               it=1,
                               out_path=args.out_path,
                               name="val_xnv",
                               wandb_act=False)

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
        "pred_time": pred_time
    }

    with open(f"{args.out_path}/results_singlemapping.json", "w") as json_file:
        json.dump(results, json_file, indent = 4)

    psnr_mean = 0.0
    ssim_mean = 0.0
    lpips_mean = 0.0
    pred_time_mean = 0.0

    for i in range(dataset.get_n_images("val")):
        xnv_val, img_val, depths_val = dataset.get_X_target("val", i, device=device)
        start_time = time.time()
        pred_rgb_tr = linear_net(embed_fn(xnv_val))
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
    with open(f"{args.out_path}/val_results_singlemapping.json", "w") as json_file:
        json.dump(results, json_file, indent = 4)