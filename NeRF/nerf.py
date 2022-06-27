import torch
import torch.nn as nn

def get_sampling_locations(rays_o, rays_d, near, far, n_samples, stratified=True):
    depths = torch.linspace(near, far, n_samples).to(rays_d)

    if stratified:
        # Generate for each ray, n_samples of uniform noise
        noise = torch.rand(list(rays_d.shape[:-1]) + [n_samples]).to(rays_d)
        
        # Limit the noise values from 0 to the length of the division
        noise *= (far-near)/n_samples
        depths = depths + noise

    # Need to broadcast the ray direction to each location
    # locations Size([n_poses, H*W, n_samples, 3])
    locations = rays_o[...,None,:] + rays_d[...,None,:] * depths[...,:,None]

    locations = torch.reshape(locations, [-1,n_samples,3])
    depths = torch.reshape(depths, [-1,n_samples])
    
    return locations, depths

def positional_encoding(input, L=6, log_sampling=False):
    if log_sampling:
        freq_bands = 2.**torch.linspace(0., L-1, L)
    else:
        freq_bands = torch.linspace(2.**0., 2.**(L-1), L)

    enc = [input]
    for f in freq_bands:
        for fn in [torch.sin, torch.cos]:
            enc.append(fn(f * input))

    return torch.cat(enc, dim=-1)

def volume_rendering(raw_radiance_density, depths, rays_d, raw_noise_std=0.):

    # distance between adjacent samples dists Size([H*W, n_samples-1])
    dists = depths[..., 1:] - depths[..., :-1]
    
    # last sample' distance is going to be infinity, we represent such distance 
    # with a symbolic high value (i.e., 1e10) dists Size([H*W, n_samples])
    inf = torch.tensor([1e10]).to(dists)
    dists = torch.concat([dists, torch.broadcast_to(inf, dists[..., :1].shape)], dim=-1).to(raw_radiance_density)
    
    
    # convert the distances in the rays to real-world distances by multiplying 
    # by the norm of the ray's direction
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    #print(dists)

    # Adding noise to model's predictions for density helps regularizing the network 
    # during training. raw_radiance_density Size([H*W, n_samples])
    #print("VOL raw_radiance_density", raw_radiance_density.shape)
    raw_density = raw_radiance_density[..., 3]
    if raw_noise_std > 0.:
        noise = torch.randn(raw_density.shape).to(raw_density) * raw_noise_std
        raw_density = raw_density + noise

    #print(raw_density)
    # alpha Size([n_images, H*W, n_samples])
    alpha = 1.0 - torch.exp(-raw_density * dists)
    #print("VOL alpha", alpha.shape)

    #print(alpha)
    # transmitance Size([n_images, H*W, n_samples])
    transmitance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    #print("VOL transmitance", transmitance.shape)
    #print(transmitance)
    # exclusive = True, for the first sample the transmitance should be 1 as the
    # probability to travel from one sample to the same sample without hitting any
    # particle is 1
    transmitance = torch.roll(transmitance, 1, -1)
    transmitance[..., 0] = 1.0

    weights = transmitance * alpha
    rgb = torch.sum(weights[..., None] * raw_radiance_density[..., :3], dim=-2)
    #print(rgb)
    return rgb, weights

def sample_pdf(locations, weights, N_samples=128, plt_fn=None, rays_o=None, rays_d=None):
    weights += 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.concat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1) # prob of being between -1 and 0 is 0
    
    u = torch.rand(list(cdf.shape[:-1]) + [N_samples]).to(cdf)

    js = torch.searchsorted(cdf, u, right=True)
    prev_j = torch.maximum(torch.tensor(0), js-1)
    j = torch.minimum(torch.tensor(cdf.shape[-1]-1), js)

    cdf_prev_j = torch.gather(cdf, dim=-1, index=prev_j)
    cdf_j = torch.gather(cdf, dim=-1, index=j)

    locations_prev_j = torch.gather(locations, dim=-1, index=prev_j)
    locations_j = torch.gather(locations, dim=-1, index=j)

    denom = cdf_j-cdf_prev_j
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom) # avoid division by 0
    t = (u-cdf_prev_j)/denom
    
    samples = locations_prev_j + t * (locations_j-locations_prev_j)

    if plt_fn is not None and rays_o is not None and rays_d is not None:
        plt_fn(pdf, cdf, j, samples)

    return samples


class NeRFModel(nn.Module):

    def __init__(self, D=8, W=256, input_ch=3, view_dir_ch=3, output_ch=4, 
                 skips=[4], viewing_direction=True, L_x=10, L_d=4, pos_enc=True):
        super(NeRFModel, self).__init__()

        self.input_ch = input_ch # size of vector x - usually 3D vector but it can have positional encoding
        self.view_dir_ch = view_dir_ch # we express the 2D viewing direction $(\theta, \phi)$ as a 3D Cartesian unit vector d
        self.output_ch = output_ch # 3 dimensions for the color value/radiance, 1 dimension for the volume density
        self.viewing_direction = viewing_direction
        self.skips = skips
        self.pos_enc = pos_enc

        if not pos_enc:
            L_x = L_d = 0

        self.L_x = L_x
        self.L_d = L_d

        volume_density_layers = [nn.Linear(in_features=input_ch + input_ch*2*L_x, out_features=W)]

        for l in range(1, D):
            volume_density_layers.append(nn.ReLU())

            if l in skips:
                volume_density_layers.append(nn.Linear(in_features=W + input_ch + input_ch*2*L_x, out_features=W))
            else:
                volume_density_layers.append(nn.Linear(in_features=W, out_features=W))

        if viewing_direction:
            self.volume_density_out = nn.Sequential(nn.Linear(in_features=W, out_features=1),
                                                    nn.ReLU())
            self.feature_vector = nn.Linear(in_features=W, out_features=W)

            radiance_layers = [nn.Linear(in_features=W + view_dir_ch + view_dir_ch*2*L_d, out_features=W//2),
                               nn.Linear(in_features=W//2, out_features=3),
                               nn.Sigmoid()]
            self.radiance_layers = nn.Sequential(*radiance_layers)

        else:
            volume_density_layers.append(nn.Linear(in_features=W, out_features=output_ch))

        self.volume_density_layers = nn.Sequential(*volume_density_layers)
    
    def forward(self, x, view_dirs=None):
        # x Size([H*W*n_images*n_samples, 3])
        #print("FWD x", x.shape)
        enc_x = positional_encoding(x, L=self.L_x)
        #print("Has nan? enc_x", utils.has_nan(enc_x))
        out_density = enc_x

        for l, layer in enumerate(self.volume_density_layers):
            out_density = layer(torch.cat((out_density, enc_x), dim=-1) if l/2 in self.skips else out_density)
        
        #print("Has nan? out_density", utils.has_nan(out_density))
        if self.viewing_direction:
            volume_density = self.volume_density_out(out_density)
            #print("Has nan? volume_density", utils.has_nan(volume_density))
            feature_vector = self.feature_vector(out_density)
            #print("Has nan? feature_vector", utils.has_nan(feature_vector))
            enc_view_dirs = positional_encoding(view_dirs, L=self.L_d)
            #print("Has nan? enc_view_dirs", utils.has_nan(enc_view_dirs))
            out_radiance = self.radiance_layers(torch.cat((feature_vector, enc_view_dirs), dim=-1))
            
        else:
            volume_density = torch.relu(out_density[..., 3])
            out_radiance = torch.sigmoid(out_density[..., :3])
        
        #print("out_radiance", out_radiance)
        #print("volume_density", volume_density)
        return torch.cat((out_radiance, volume_density), dim=-1)