import lpips
from functools import reduce
import numpy as np
import open3d.core as o3c

import torch
import torch.nn.functional as F
from tqdm import tqdm


import nvidia_smi
nvidia_smi.nvmlInit()

def print_memory_usage():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"Device {0}: {nvidia_smi.nvmlDeviceGetName(handle)}, Memory : ({100*info.free/info.total:.2f}% free): {info.total} (total), {info.free} (free), {info.used} (used)")


def append_dict(dict, new_dict):
    for key in dict:
        dict[key].append(new_dict[key])
    return dict


def has_nan(tensor):
    return tensor.isnan().any()


def has_inf(tensor):
    return tensor.isinf().any()


def summarize_diff(old_arr, new_arr):
    assert(old_arr.shape == new_arr.shape,
           "The 2 arrays have different shape, cannot find the differences.")

    total_items = reduce(lambda x, y: x*y, list(new_arr.shape))
    diff_items = np.where(old_arr != new_arr)
    num_diff = np.count_nonzero(old_arr != new_arr)

    print(
        f"\tChanged {num_diff} items out of {total_items} in array of shape {new_arr.shape}")
    if num_diff > 0:
        print(
            f"\t\tFrom ({diff_items[0][0]}, {diff_items[1][0]}, {diff_items[2][0]}) to ({diff_items[0][-1]}, {diff_items[1][-1]}, {diff_items[2][-1]})")

####################       RAY CASTING       ####################


def torch2open3d(torch_tensor):
    return o3c.Tensor(torch_tensor.cpu().numpy())


def cast_rays(scene, rays_od):
    ans = scene.cast_rays(torch2open3d(rays_od.float()))
    hit = ans['t_hit'].numpy()
    return hit

####################       LINEAR MAPPING       ####################


def filter_duplicates(X, batch_size=1_000_000):
    X_unique = None
    tqdm._instances.clear()
    for i in tqdm(range(0, X.shape[0], batch_size), unit="batch", leave=False, desc=f"Filtering {X.shape[0]} points on {X.device}"):
        X_batch, inverse = torch.unique(
            X[i:i+batch_size], sorted=True, return_inverse=True, dim=0)
        perm = torch.arange(inverse.size(
            0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        indices_batch = inverse.new_empty(
            X_batch.size(0)).scatter_(0, inverse, perm)
        indices_batch += i

        if X_unique is None:
            X_unique, indices = X_batch, indices_batch
        else:
            xh_unique_tmp = torch.cat([X_unique, X_batch])
            indices = torch.cat([indices, indices_batch])

            X_unique, inverse = torch.unique(
                xh_unique_tmp, return_inverse=True, dim=0)
            perm = torch.arange(inverse.size(
                0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            indices_nonrep = inverse.new_empty(
                X_unique.size(0)).scatter_(0, inverse, perm)
            indices = indices[indices_nonrep]

    print(f"Filtered {X.shape[0]} points to {X_unique.shape[0]}")
    return X_unique, indices


def compute_ssim(img0,
                 img1,
                 max_val=1.0,
                 filter_size=11,
                 filter_sigma=1.5,
                 k1=0.01,
                 k2=0.03,
                 return_map=False):

    device = img0.device
    ori_shape = img0.size()
    width, height, num_channels = ori_shape[-3:]
    img0 = img0.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    img1 = img1.view(-1, width, height, num_channels).permute(0, 3, 1, 2)
    batch_size = img0.shape[0]

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((torch.arange(filter_size, device=device) -
           hw + shift) / filter_sigma) ** 2
    filt = torch.exp(-0.5 * f_i)
    filt /= torch.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    # z is a tensor of size [B, H, W, C]
    def filt_fn1(z): return F.conv2d(
        z, filt.view(1, 1, -1, 1).repeat(num_channels, 1, 1, 1),
        padding=[hw, 0], groups=num_channels)

    def filt_fn2(z): return F.conv2d(
        z, filt.view(1, 1, 1, -1).repeat(num_channels, 1, 1, 1),
        padding=[0, hw], groups=num_channels)

    # Vmap the blurs to the tensor size, and then compose them.
    def filt_fn(z): return filt_fn1(filt_fn2(z))
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0 ** 2) - mu00
    sigma11 = filt_fn(img1 ** 2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = torch.clamp(sigma00, min=0.0)
    sigma11 = torch.clamp(sigma11, min=0.0)
    sigma01 = torch.sign(sigma01) * torch.min(
        torch.sqrt(sigma00 * sigma11), torch.abs(sigma01)
    )

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = torch.mean(ssim_map.reshape([-1, num_channels*width*height]), dim=-1)
    return ssim_map if return_map else ssim.item()


def compute_lpips(img0, img1, lpips, device=torch.device("cuda")):
    with torch.no_grad():
        if device == torch.device("cuda"):
            return lpips(img0.permute([2, 0, 1]).cuda().contiguous(),
                            img1.permute([2, 0, 1]).cuda().contiguous(),
                            normalize=True).item()
        
        else:
            return lpips(img0.permute([2, 0, 1]).contiguous(),
                            img1.permute([2, 0, 1]).contiguous(),
                            normalize=True).item()
