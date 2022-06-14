from xmlrpc.client import boolean
import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import seaborn as sn
import utils.visualization as v
import utils.data as data
import utils.utils as utils
import nerf
import configargparse
import wandb

def run_model(locations, view_dirs, model, n_samples, chunk_size, device=torch.device("cuda")):
    # locations Size([H*W*n_images, n_samples, 3])
    locs_flat = torch.reshape(locations, [-1, locations.shape[-1]])
    
    # view_dirs Size([n_images, H*W, 1, 3])
    view_dirs = torch.broadcast_to(view_dirs, [view_dirs.shape[0], view_dirs.shape[1], n_samples, locations.shape[-1]])
    # view_dirs Size([n_images, H*W, n_samples, 3])
    view_dirs_flat = torch.reshape(view_dirs, [-1, view_dirs.shape[-1]])

    raw_radiance_density_flat = model(locs_flat, view_dirs=view_dirs_flat)
    raw_radiance_density = torch.reshape(raw_radiance_density_flat, 
                                            list(locations.shape[:-1]) + [raw_radiance_density_flat.shape[-1]])
    '''raw_radiance_density = None

    for i in range(0, locs_flat.shape[0], chunk_size):
        raw_radiance_density_flat = model(locs_flat[i:i+chunk_size], 
                                          view_dirs=view_dirs_flat[i:i+chunk_size])
        
        #print("Has nan? raw_radiance_density_flat", utils.has_nan(raw_radiance_density_flat))
        raw_radiance_density_chunk = torch.reshape(raw_radiance_density_flat, 
                                            [-1] + list(locations.shape[1:-1]) + [raw_radiance_density_flat.shape[-1]])
        if raw_radiance_density is None:
            raw_radiance_density = raw_radiance_density_chunk
        else:
            raw_radiance_density = torch.cat((raw_radiance_density, raw_radiance_density_chunk), dim=0)'''

    return raw_radiance_density

def predict(batch_rays, model, chunk_size=1024*32, N_samples=64, model_f=None, N_f=128, near=2, far=6, raw_noise_std=0.0, device=torch.device("cuda")):
    rays_o = batch_rays[..., :3]
    rays_d = batch_rays[..., 3:6]
    view_dirs = batch_rays[..., None, 6:9]
    
    # locations Size([n_poses*H*W, n_samples, 3])
    locations, depths = nerf.get_sampling_locations(rays_o=rays_o, 
                                                   rays_d=rays_d, 
                                                   near=near,
                                                   far=far,
                                                   n_samples=N_samples)
    
    print("Has nan? locations", utils.has_nan(locations))
    print("Has nan? depths", utils.has_nan(depths))
    raw_radiance_density = run_model(locations, view_dirs, model, N_samples, chunk_size, device=device)
    print("Has nan? raw_radiance_density", utils.has_nan(raw_radiance_density))
    rgb_map, weights = nerf.volume_rendering(raw_radiance_density, depths, 
                                        rays_d, raw_noise_std=raw_noise_std)
    print("Has nan? rgb_map", utils.has_nan(rgb_map))
    
    if N_f > 0:
        rgb_map_0, weights_0 = rgb_map, weights

        depths_mid = .5 * (depths[..., 1:] + depths[..., :-1])
        depths_f = nerf.sample_pdf(depths_mid, weights[..., 1:-1])
        depths_f = depths_f.detach()
        
        depths_new = torch.cat((depths, depths_f), dim=-1)
        depths_new, _ = torch.sort(depths_new, dim=-1)
        locs_f = rays_o[...,None,:] + rays_d[...,None,:] * depths_new[...,:,None]

        raw_radiance_density = run_model(locs_f, view_dirs, model_f)
        rgb_map, weights = nerf.volume_rendering(raw_radiance_density, depths_new, 
                                            rays_d, raw_noise_std=raw_noise_std)

    pred = {'rgb_map': rgb_map, 'weights': weights}
    if N_f > 0:
        pred['rgb_map_0'] = rgb_map_0
        pred['weights_0'] = weights_0
    return pred

def parse_args():
    parser = configargparse.ArgumentParser(description="Initializes the geometry with a given mesh")
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--mesh', type=str, help='initial mesh path')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--out_path', type=str, help='path to the output folder', default="./out")
    parser.add_argument('--dataset_type', type=str, help='type of dataset', choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--batch_size', type=int, default=1, help='number of images whose rays would be used at once')
    parser.add_argument('--chunk_size', type=int, default=1024*32)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--N_samples', type=int, help='number of points to sample with the coarse network', default=64)
    parser.add_argument('--D_c', type=int, help='Depth coarse network', default=8)
    parser.add_argument('--W_c', type=int, help='Coarse network width', default=256)
    parser.add_argument('--N_f', type=int, help='number of points to sample with the fine network', default=128)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--N_iters', type=int, help='number of iterations to train the network', default=1000)
    parser.add_argument('--near', type=int, help='near distance to the camera where to start sampling the rays', default=2)
    parser.add_argument('--far', type=int, help='far distance to the camera where to start sampling the rays', default=6)
    parser.add_argument('--raw_noise_std', type=float, default=1.0)
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print("2", args.dataset_to_gpu)

    wandb.init(project="controllable-neural-rendering", entity="guillemgarrofe")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
     
    # Load data
    print("dataset to: ", device if args.dataset_to_gpu == True else torch.device("cpu"))
    dataset = data.NeRFDataset(args, device=device if args.dataset_to_gpu else torch.device("cpu"))
    
    wandb.config = {
        "learning_rate": args.lr,
        "num_iters": args.N_iters,
        "batch_size": dataset.hwf[0]*dataset.hwf[1] if args.batch_size is None else args.batch_size,
        "n_f": args.N_f,
        "dataset_type": args.dataset_type,
        "dataset_path": args.dataset_path 
    }

    # Create models
    print("Creating coarse model...")
    model = nerf.NeRFModel(D=args.D_c)
    model.to(device)
    model_f = None

    if args.N_f > 0:
        print("Creating fine model...")
        model_f = nerf.NeRFModel(D=2)
        model_f.to(device)

    # Create optimizer
        print("Creating optimizer...")
        optimizer = torch.optim.Adam(list(model.parameters()) + list(model_f.parameters()), lr=args.lr)

    else:
        print("Creating optimizer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    #loss_fn = nn.MSELoss()
    img2mse = lambda x, y : torch.mean((x - y) ** 2)
    
    # Train
    training_losses = []
    val_losses = []
    
    for it in tqdm(range(args.N_iters), unit="iteration"):
        # By default each batch will correspond to the rays of a single image
        batch_rays_tr, target_rgb_tr = dataset.next_batch("train", device=device)
        print("Has nan? batch_rays", utils.has_nan(batch_rays_tr))
        model.train()
        pred = None
        #for i in range(0, batch_rays_tr.shape[0], args.chunk_size):
        pred = predict(batch_rays_tr, 
                        model, 
                        chunk_size=args.chunk_size,
                        N_samples=args.N_samples, 
                        model_f=model_f, 
                        N_f=args.N_f, 
                        near=args.near, 
                        far=args.far, 
                        raw_noise_std=args.raw_noise_std,
                        device=device)

        '''if pred is None:
            pred = chunk_pred
        else:
            pred = append_dict(pred, chunk_pred)'''

        if args.N_f > 0:
            loss_c = img2mse(pred['rgb_map_0'], target_rgb_tr)
            loss_f = img2mse(pred['rgb_map'], target_rgb_tr)
            loss = loss_c + loss_f
            wandb.log({
                "loss_tr": loss,
                "loss_c_tr": loss_c,
                "loss_f_tr": loss_f,
                "gpu_memory": torch.cuda.memory_allocated(0)/1024/1024/1024
                })
        else:
            print(pred['rgb_map'].shape, target_rgb_tr.shape)
            loss = img2mse(pred['rgb_map'], target_rgb_tr)
            print("loss", loss)
            wandb.log({"loss": loss})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())

        # ----- VALIDATION LOSS -----
        model.eval()
        batch_rays_val, target_rgb_val = dataset.next_batch("val", device=device)
        pred = predict(batch_rays_val, model, N_samples=args.N_samples, model_f=model_f, N_f=args.N_f, near=args.near, far=args.far)
        
        if args.N_f > 0:
            loss_c = img2mse(pred['rgb_map_0'], target_rgb_val)
            loss_f = img2mse(pred['rgb_map'], target_rgb_val)
            loss_val = loss_c + loss_f
            wandb.log({
                "loss_val": loss_val,
                "loss_c_val": loss_c,
                "loss_f_val": loss_f
                })
        else:
            loss_val = img2mse(pred['rgb_map'], target_rgb_val)
            wandb.log({"loss_val": loss_val})

        val_losses.append(loss_val.item())
        
        if it%500 == 0:
            v.validation_view(pred['rgb_map'].detach().cpu(), target_rgb_val.detach().cpu(), img_shape=(dataset.hwf[0], dataset.hwf[1], 3), it=it, out_path=args.out_path)
            #v.plot_losses(training_losses, val_losses, it=it)
