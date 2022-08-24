import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import configargparse
import wandb

import open3d as o3d
import lpips

import visualization as v
import sys
import time
sys.path.append('../')

def parse_args():
    parser = configargparse.ArgumentParser(description="Initializes the geometry with a given mesh")
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--mesh_path', type=str, help='initial mesh path')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--out_path', type=str, help='path to the output folder', default="./out")
    
    parser.add_argument('--dataset_type', type=str, help='type of dataset', choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--encoding_freqs', type=int, help='number of frequencies used in the positional encoding', default=6)
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    parser.add_argument('--load_light', action='store_true', help='load light sources positions')
    parser.add_argument("--test", action='store_true', help='use reduced number of images')

    parser.add_argument('--batch_size', type=int, default=200_000, help='number of points whose rays would be used at once')
    parser.add_argument('--num_iters', type=int, help='number of iterations to train the network', default=1000)
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
    
    args = parser.parse_args()
    return args


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
    dataset = data.NeRFDataset(args)
    dataset.switch_2_xnv_dataset()

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
                        "dataset_type": args.dataset_type,
                        "dataset_path": args.dataset_path 
                    })

    # Create models
    embed_fn, input_ch = emb.get_posenc_embedder(in_dim=6, num_freqs=args.encoding_freqs)
    rend_net = net.SurfaceRenderingNetwork(input_ch=input_ch+3,
                                           out_ch=3,
                                           hidden_ch=[512, 512, 512, 512],
                                           embed_fn=embed_fn)
    rend_net.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(rend_net.parameters(), lr=args.lrate)

    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

    iter = 0
    if wandb.run.resumed:
        wandb.restore(f"{args.checkpoint_path}/{args.run_id}.tar")
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.run_id}.tar")
        rend_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iter = checkpoint['iter']
        print(f"Resuming {run.id} at iteration {iter}")

    pbar = tqdm(total=args.num_iters, unit="iteration")
    pbar.update(iter)
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
    while iter < args.num_iters:
        batch_xnv_tr, target_rgb_tr = dataset.next_batch("train", device=device)
        
        rend_net.train()
        pred_rgb = rend_net(batch_xnv_tr[..., :3], batch_xnv_tr[..., 3:6], batch_xnv_tr[..., 6:])
        
        loss = loss_fn(pred_rgb, target_rgb_tr)

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
                }, step=iter)

        #  --------------- VALIDATION --------------

        rend_net.eval()
        batch_xnv_val, target_rgb_val = dataset.next_batch("val", device=device)
        pred_rgb = rend_net(batch_xnv_val[..., :3], batch_xnv_val[..., 3:6], batch_xnv_val[..., 6:])

        val_loss = loss_fn(pred_rgb, target_rgb_val)

        wandb.log({
                "val_loss": val_loss,
                "val_psnr": mse2psnr(val_loss)
                }, step=iter)

        # --------------- EVALUATION --------------
        if (iter < 200 and iter%20 == 0) or iter%2000 == 0:
            xnv, img, depths = dataset.get_X_target("train", 0, device=device)
            rgb_map = None
            for i in range(0, xnv.shape[0], args.batch_size):
                pred = rend_net(xnv[i:i+args.batch_size, :3], xnv[i:i+args.batch_size, 3:6], xnv[i:i+args.batch_size, 6:])
                
                if rgb_map is None:
                    rgb_map = pred
                else:
                    rgb_map = torch.cat((rgb_map, pred), dim=0)

            v.validation_view_rgb_xndv(rgb_map.detach().cpu(), 
                                      img.detach().cpu(), 
                                      points=xnv[..., :3].detach().cpu(),
                                      normals=xnv[..., 3:6].detach().cpu(),
                                      depths=depths,
                                      viewdirs=xnv[..., 6:].detach().cpu(),
                                      img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                      it=iter, 
                                      out_path=args.out_path,
                                      name="training_xnv")

            xnv, img, depths = dataset.get_X_target("train", np.random.randint(0, dataset.get_n_images()), device=device)
            rgb_map = None
            for i in range(0, xnv.shape[0], args.batch_size):
                pred = rend_net(xnv[i:i+args.batch_size, :3], xnv[i:i+args.batch_size, 3:6], xnv[i:i+args.batch_size, 6:])
                
                if rgb_map is None:
                    rgb_map = pred
                else:
                    rgb_map = torch.cat((rgb_map, pred), dim=0)

            v.validation_view_rgb_xndv(rgb_map.detach().cpu(), 
                                      img.detach().cpu(), 
                                      points=xnv[..., :3].detach().cpu(),
                                      normals=xnv[..., 3:6].detach().cpu(),
                                      depths=depths,
                                      viewdirs=xnv[..., 6:].detach().cpu(),
                                      img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                      it=iter, 
                                      out_path=args.out_path,
                                      name="training_random_xnv")
            
            img_shape=(dataset.hwf[0], dataset.hwf[1], 3)
            wandb.log({
                "tr_ssim": utils.compute_ssim(torch.reshape(img, img_shape), 
                                              torch.reshape(rgb_map, img_shape)),
                "tr_lpips": utils.compute_lpips(torch.reshape(img, img_shape), 
                                                torch.reshape(rgb_map, img_shape),
                                                lpips_vgg,
                                                device)
                }, step=iter)

            xnv, img, depths = dataset.get_X_target("val", 0, device=device)
            rgb_map = None
            for i in range(0, xnv.shape[0], args.batch_size):
                pred = rend_net(xnv[i:i+args.batch_size, :3], xnv[i:i+args.batch_size, 3:6], xnv[i:i+args.batch_size, 6:])
                
                if rgb_map is None:
                    rgb_map = pred
                else:
                    rgb_map = torch.cat((rgb_map, pred), dim=0)

            v.validation_view_rgb_xndv(rgb_map.detach().cpu(), 
                                      img.detach().cpu(), 
                                      points=xnv[..., :3].detach().cpu(),
                                      normals=xnv[..., 3:6].detach().cpu(),
                                      depths=depths,
                                      viewdirs=xnv[..., 6:].detach().cpu(),
                                      img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                      it=iter, 
                                      out_path=args.out_path)

            xnv, img, depths = dataset.get_X_target("val", np.random.randint(0, dataset.get_n_images("val")), device=device)
            rgb_map = None
            for i in range(0, xnv.shape[0], args.batch_size):
                pred = rend_net(xnv[i:i+args.batch_size, :3], xnv[i:i+args.batch_size, 3:6], xnv[i:i+args.batch_size, 6:])
                
                if rgb_map is None:
                    rgb_map = pred
                else:
                    rgb_map = torch.cat((rgb_map, pred), dim=0)

            v.validation_view_rgb_xndv(rgb_map.detach().cpu(), 
                                      img.detach().cpu(), 
                                      points=xnv[..., :3].detach().cpu(),
                                      normals=xnv[..., 3:6].detach().cpu(),
                                      depths=depths,
                                      viewdirs=xnv[..., 6:].detach().cpu(),
                                      img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                      it=iter, 
                                      out_path=args.out_path,
                                      name="val_random_xnv")

            ssip_mean = 0.0
            lpips_mean = 0.0
            for i in range(dataset.get_n_images("val")):
                xnv, img, depths = dataset.get_X_target("val", i, device=device)

                start_time = time.time()
                rgb_map = None
                for i in range(0, xnv.shape[0], args.batch_size):
                    pred = rend_net(xnv[i:i+args.batch_size, :3], xnv[i:i+args.batch_size, 3:6], xnv[i:i+args.batch_size, 6:])
                    
                    if rgb_map is None:
                        rgb_map = pred
                    else:
                        rgb_map = torch.cat((rgb_map, pred), dim=0)
                pred_time = time.time() - start_time

                ssip_val = utils.compute_ssim(torch.reshape(img, img_shape), 
                                               torch.reshape(rgb_map, img_shape))
                lpips_val = utils.compute_lpips(torch.reshape(img, img_shape), 
                                                 torch.reshape(rgb_map, img_shape),
                                                 lpips_vgg,
                                                 device)
                psnr_val = mse2psnr(loss_fn(rgb_map.to(device), pred_rgb))

                ssip_mean = (ssip_mean*i + ssip_val)/(i+1)
                lpips_mean = (lpips_mean*i + lpips_val)/(i+1)
                psnr_mean = (psnr_mean*i + psnr_val)/(i+1)
                pred_time_mean = (pred_time_mean*i + pred_time)/(i+1)

            wandb.log({
                "val_ssim": ssip_mean,
                "val_lpips": lpips_mean
                }, step=iter)

            torch.save({ # Save our checkpoint loc
                'iter': iter,
                'model_state_dict': rend_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, f"{args.checkpoint_path}/{run.id}.tar")
            wandb.save(f"{args.checkpoint_path}/{run.id}.tar") # saves checkpoint to wandb    

        iter += 1
        pbar.update(1)
    
    pbar.close()