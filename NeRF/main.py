import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import tqdm
import seaborn as sn
import utils.visualization as v
import utils.data as data
import nerf
import configargparse

def run_model(locations, view_dirs, model):
    locs_flat = torch.reshape(locations, [-1, locations.shape[-1]])
    
    view_dirs = torch.broadcast_to(view_dirs, locations.shape)
    view_dirs_flat = torch.reshape(view_dirs, [-1, view_dirs.shape[-1]])

    raw_radiance_density_flat = model(locs_flat, view_dirs=view_dirs_flat)
    raw_radiance_density = torch.reshape(raw_radiance_density_flat, 
                                            list(locations.shape[:-1]) + [raw_radiance_density_flat.shape[-1]])
    return raw_radiance_density

def predict(batch_rays, model, N_samples=64, model_f=None, N_f=128, near=2, far=6, raw_noise_std=0.0):
    rays_o = batch_rays[..., :3]
    rays_d = batch_rays[..., 3:6]
    view_dirs = batch_rays[..., None, 6:9]
    locations, depths = nerf.get_sampling_locations(rays_o=rays_o, 
                                                   rays_d=rays_d, 
                                                   near=near, 
                                                   far=far, 
                                                   n_samples=N_samples)
    
    raw_radiance_density = run_model(locations, view_dirs, model)
    rgb_map, weights = nerf.volume_rendering(raw_radiance_density, depths, 
                                        rays_d, raw_noise_std=raw_noise_std)
    
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
    parser.add_argument('--out_path', type=str, help='path to the output folder')
    parser.add_argument('--dataset_type', type=str, help='type of dataset', choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--batch_size', type=int,)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--N_f', type=int, )
    parser.add_argument('--lr', type=float, default=2e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    device = "cpu" #torch.device("cuda"  if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
     
    # Load data
    dataset = data.NeRFDataset(args, device=device)

    # Create models
    model = nerf.NeRFModel()
    model.to(device)
    model_f = None

    if args.N_f > 0:
        model_f = nerf.NeRFModel(D=2)
        model_f.to(device)

    # Create optimizer
        optimizer = torch.optim.Adam(list(model.parameters()) + list(model_f.parameters()), lr=lr)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    loss_fn = nn.MSELoss()

    # Train
    train_iterator = iter(loader_train)
    val_iterator = iter(loader_val)
    training_losses = []
    val_losses = []

    for it in tqdm(range(N_iters), unit="iteration"):
        try:
            batch_rays_tr, target_rgb = next(train_iterator)
        except StopIteration:
            train_iterator = iter(loader_train)
            batch_rays_tr, target_rgb = next(train_iterator)

        try:
            batch_rays_val, target_rgb_val = next(val_iterator)
        except StopIteration:
            val_iterator = iter(loader_val)
            batch_rays_val, target_rgb_val = next(val_iterator)

        model.train()
        pred = predict(batch_rays_tr, model, N_samples=N_samples, model_f=model_f, 
                       N_f=N_f, near=near, far=far, raw_noise_std=raw_noise_std)

        if N_f > 0:
            loss_c = loss_fn(pred['rgb_map_0'], target_rgb)
            loss_f = loss_fn(pred['rgb_map'], target_rgb)
            loss = loss_c + loss_f
        else:
            loss = loss_fn(pred['rgb_map'], target_rgb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_losses.append(loss.item())

        # ----- VALIDATION LOSS -----
        model.eval()
        pred = predict(batch_rays_val, model, N_samples=N_samples, model_f=model_f, N_f=N_f, near=near, far=far)
        
        if N_f > 0:
            loss_c = loss_fn(pred['rgb_map_0'], target_rgb_val)
            loss_f = loss_fn(pred['rgb_map'], target_rgb_val)
            loss_val = loss_c + loss_f
        else:
            loss_val = loss_fn(pred['rgb_map'], target_rgb_val)

        val_losses.append(loss_val.item())
        
        if it%500 == 0:
            validation_view(model, poses[i_val[0]], hwf, images[i_val[0]], it=it, img_shape=images.shape[-3:], 
                            N_samples=N_samples, model_f=model_f, N_f=N_f, near=near, far=far)
            v.plot_losses(training_losses, val_losses, it=it)
