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
from PIL import Image
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
    parser.add_argument('--load_light', action='store_true', help='load light sources positions')
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

    parser.add_argument('--num_voxels', type=int,
                        help='number of voxels that contain surface points. The cubic root has to be an integer.', default=27)

    args = parser.parse_args()
    return args

def compute_inv(xnv, target, boundaries, embed_fn, device=torch.device("cuda"), gpu_limit=1e07):
    x_bound, y_bound, z_bound = boundaries
    mask =  (x_bound[0] <= xnv[..., 0]) & (xnv[..., 0] <= x_bound[1]) & \
            (y_bound[0] <= xnv[..., 1]) & (xnv[..., 1] <= y_bound[1]) & \
            (z_bound[0] <= xnv[..., 2]) & (xnv[..., 2] <= z_bound[1])

    xnv, target = xnv[mask], target[mask]
    if xnv.shape[0] >= gpu_limit:
        xnv, indices = utils.filter_duplicates(xnv)
        target = target[indices]

    if xnv.shape[0] < gpu_limit:
        xnv_enc_inv = torch.linalg.pinv(embed_fn(xnv.to(device)))
        linear_mapping = xnv_enc_inv @ target.to(device)
    else:
        xnv_enc_inv = torch.linalg.pinv(embed_fn(xnv.cpu()))
        linear_mapping = xnv_enc_inv @ target.cpu()
        
    return linear_mapping.T.to(device)


def get_voxel_ids(X, boundaries):
    voxel_ids = torch.zeros((0, 1), dtype=torch.long).to(X.device)
    row_ids = torch.zeros((0, 1), dtype=torch.long).to(X.device)
    for i, boundary in enumerate(boundaries):
        x_bound, y_bound, z_bound = boundary
        mask =  (x_bound[0] <= X[..., 0]) & (X[..., 0] <= x_bound[1]) & \
                (y_bound[0] <= X[..., 1]) & (X[..., 1] <= y_bound[1]) & \
                (z_bound[0] <= X[..., 2]) & (X[..., 2] <= z_bound[1])
        
        voxel_ids = torch.cat((voxel_ids, torch.ones((torch.count_nonzero(mask), 1), dtype=torch.long).to(X.device)*i))
        row_ids = torch.cat((row_ids, torch.nonzero(mask)))
        
    return torch.squeeze(row_ids), torch.squeeze(voxel_ids)

def my_ceil(a, precision=0):
    return torch.true_divide(torch.ceil(a * 10**precision), 10**precision)

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
    if not args.only_eval:
        xnv, target_rgb, depths = dataset.get_tensors("train", device=device)

        # A cube can be split in n^3 small voxels. 
        voxels_per_axis = np.cbrt(args.num_voxels)

        linear_mappings = torch.zeros([args.num_voxels, 3, input_ch]).to(device)
        mask = (xnv[:, 0] == -1.) & (xnv[:, 1] == -1.) & (xnv[:, 2] == -1.)  # not masking takes too much time
        x_min, x_max = torch.min(xnv[~mask, 0]), torch.max(xnv[~mask, 0])
        y_min, y_max = torch.min(xnv[~mask, 1]), torch.max(xnv[~mask, 1])
        z_min, z_max = torch.min(xnv[~mask, 2]), torch.max(xnv[~mask, 2])
        x_step = my_ceil((x_max-x_min)/voxels_per_axis, precision=4)
        y_step = my_ceil((y_max-y_min)/voxels_per_axis, precision=4)
        z_step = my_ceil((z_max-z_min)/voxels_per_axis, precision=4)
        range_x = torch.arange(x_min, x_max, x_step)
        range_y = torch.arange(y_min, y_max, y_step)
        range_z = torch.arange(z_min, z_max, z_step)
        start_time = time.time()

        i_voxel = 0
        boundaries = []
        
        for i, x in enumerate(range_x):
            for j, y in enumerate(range_y):
                for k, z in enumerate(range_z):
                    boundary = ((x, x+x_step), (y, y+y_step), (z, z+z_step))
                    linear_mappings[i_voxel] = compute_inv(xnv, 
                                                            target_rgb,
                                                            boundaries=boundary,
                                                            embed_fn=embed_fn,
                                                            device=device)
                    boundaries.append(boundary)
                    i_voxel += 1
        train_time = time.time() - start_time
        print("Training time: %s seconds." % (train_time))

        xnv_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)
        row_ids, voxel_ids = get_voxel_ids(xnv_tr, boundaries)
        v.plot_voxels_3Dpoints(xnv_tr[..., :3], row_ids, voxel_ids,
                                colab=args.colab, out_path=args.out_path, filename="train_voxels.png")

    else:
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.num_voxels}voxels.tar")
        linear_mappings = checkpoint['linear_mappings']
        boundaries = checkpoint['boundaries']


    linear_net = net.VoxelisedLinearNetwork(in_features=input_ch, 
                                        linear_mappings=linear_mappings, 
                                        num_freqs=6)
    linear_net.to(device)

    # EVALUATION
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
    img_shape = (dataset.hwf[0], dataset.hwf[1], 3)
    xnv_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)

    start_time = time.time()
    row_ids, voxel_ids = get_voxel_ids(xnv_tr, boundaries)
    pred_rgb_tr = linear_net(xnv_tr, row_ids, voxel_ids)
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
                               it=args.num_voxels,
                               out_path=args.out_path,
                               name="training_xnv",
                               wandb_act=False)

    xnv_val, img_val, depths_val = dataset.get_X_target("val", 0, device=device)
    row_ids, voxel_ids = get_voxel_ids(xnv_val, boundaries)
    pred_rgb_val = linear_net(xnv_val, row_ids, voxel_ids)

    loss_val = loss_fn(pred_rgb_val, img_val)

    v.validation_view_rgb_xndv(pred_rgb_val.detach().cpu(),
                               img_val.detach().cpu(),
                               points=xnv_val[..., :3].detach().cpu(),
                               normals=xnv_val[..., 3:6].detach().cpu(),
                               depths=depths_val,
                               viewdirs=xnv_val[..., 6:].detach().cpu(),
                               img_shape=(dataset.hwf[0], dataset.hwf[1], 3),
                               it=args.num_voxels,
                               out_path=args.out_path,
                               name="val_xnv",
                               wandb_act=False)

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
            "pred_time": pred_time
        }

        with open(f"{args.out_path}/results_{args.num_voxels}voxels.json", "w") as json_file:
            json.dump(results, json_file, indent = 4)

        torch.save({ # Save our checkpoint loc
            'num_voxels': args.num_voxels,
            'linear_mappings': linear_mappings,
            'boundaries': boundaries
            }, f"{args.checkpoint_path}/{args.num_voxels}voxels.tar")

    elif dataset.get_n_images("val") > 1:
        psnr_mean = 0.0
        ssim_mean = 0.0
        lpips_mean = 0.0
        pred_time_mean = 0.0

        for i in range(dataset.get_n_images("val")):
            xnv_val, img_val, depths_val = dataset.get_X_target("val", i, device=device)
            start_time = time.time()
            row_ids, voxel_ids = get_voxel_ids(xnv_val, boundaries)
            pred_rgb_tr = linear_net(xnv_val, row_ids, voxel_ids)
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
        with open(f"{args.out_path}/val_results_{args.num_voxels}voxels.json", "w") as json_file:
            json.dump(results, json_file, indent = 4)

    elif dataset.get_n_images("test") > 0:

        for i in range(dataset.get_n_images("test")):
            xnv_test, img_test, _ = dataset.get_X_target("test", i, device=device)
            row_ids, voxel_ids = get_voxel_ids(xnv_test, boundaries)
            pred_rgb_test = linear_net(xnv_test, row_ids, voxel_ids)
            pred_rgb_test = torch.reshape(torch.clamp(pred_rgb_test, min=0.0, max=1.0), img_shape)
            im = Image.fromarray((pred_rgb_test.detach().cpu().numpy() * 255).astype(np.uint8))
            im.save(f"{args.out_path}/test/pred_{i}.png")