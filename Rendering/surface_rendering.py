import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

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
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.colab:
        sys.path.append(args.colab_path)
    import utils.data as data
    import utils.networks as net
    import utils.utils as utils
   
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

    if args.test:
        points = dataset.get_points("train", 2, device=torch.device("cpu"))
        pcd2 = o3d.t.geometry.PointCloud(utils.torch2open3d(points))
        points = dataset.get_points("train", 6, device=torch.device("cpu"))
        pcd6 = o3d.t.geometry.PointCloud(utils.torch2open3d(points))
        points = dataset.get_points("train", 5, device=torch.device("cpu"))
        pcd5 = o3d.t.geometry.PointCloud(utils.torch2open3d(points))
        o3d.visualization.draw_geometries([pcd2.to_legacy(), pcd6.to_legacy(), pcd5.to_legacy()])

        for i in range(2,3):
            #depths = dataset.get_depths("train", i, device=torch.device("cpu"))
            #normals = dataset.get_normals("train", i, device=torch.device("cpu"))
            #img = dataset.get_image("train", i, device=torch.device("cpu"))
            #v.print_depths(depths, img, dataset.hwf, args.out_path+f"/depths_{i}.png")
            #v.print_normals(normals, img, dataset.hwf, args.out_path+f"/norms_{i}.png")
            points = dataset.get_points("train", i, device=torch.device("cpu"))
            pcd = o3d.t.geometry.PointCloud(utils.torch2open3d(points))

            points = points.reshape(dataset.hwf[0], dataset.hwf[1], 3)
            depths = dataset.get_depths("train", i, device=torch.device("cpu"))
            depths = depths.reshape(dataset.hwf[0], dataset.hwf[1])
            img = dataset.get_image("train", i, device=torch.device("cpu"))
            img = img.reshape(dataset.hwf[0], dataset.hwf[1], 3)

            plt.figure(figsize=(15, 4))
            plt.subplot(131)
            plt.imshow(img[20:35,60:80,...].numpy())
            plt.title(f"Image")
            plt.subplot(132)
            plt.imshow(points[20:35,60:80,...].numpy()/10.0)
            plt.title("3D points")
            plt.subplot(133)
            plt.imshow(depths[20:35,60:80,...].numpy())
            plt.title("Depths")
            plt.show()
            plt.savefig(f"{args.out_path}/test.png")

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
                        "batch_size": dataset.hwf[0]*dataset.hwf[1] if args.batch_size is None else args.batch_size,
                        "dataset_type": args.dataset_type,
                        "dataset_path": args.dataset_path 
                    })

    # Create models
    rend_net = net.SurfaceRenderingNetwork(input_ch=9,
                                           out_ch=3,
                                           hidden_ch=[512, 512, 512, 512],
                                           enc_dim=6)
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

    pbar = tqdm(total=args.N_iters, unit="iteration")
    pbar.update(iter)
    while iter < args.N_iters:
        # By default each batch will correspond to the rays of a single image
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
                "loss": loss,
                "psnr": mse2psnr(loss)
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
        if iter%2000 == 0:
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