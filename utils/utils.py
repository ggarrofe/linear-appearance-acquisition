from functools import reduce
import numpy as np
import open3d.core as o3c

import torch
from tqdm import tqdm

def append_dict(dict, new_dict):
    for key in dict:
        dict[key].append(new_dict[key])
    return dict

def has_nan(tensor):
    return tensor.isnan().any()

def has_inf(tensor):
    return tensor.isinf().any()

def summarize_diff(old_arr, new_arr):
    assert(old_arr.shape == new_arr.shape, "The 2 arrays have different shape, cannot find the differences.")
    
    total_items = reduce(lambda x, y: x*y, list(new_arr.shape))
    diff_items = np.where(old_arr != new_arr)
    num_diff = np.count_nonzero(old_arr != new_arr)

    print(f"\tChanged {num_diff} items out of {total_items} in array of shape {new_arr.shape}")
    if num_diff > 0:
        print(f"\t\tFrom ({diff_items[0][0]}, {diff_items[1][0]}, {diff_items[2][0]}) to ({diff_items[0][-1]}, {diff_items[1][-1]}, {diff_items[2][-1]})")

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
        X_batch, inverse = torch.unique(X[i:i+batch_size], sorted=True, return_inverse=True, dim=0)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        indices_batch = inverse.new_empty(X_batch.size(0)).scatter_(0, inverse, perm)
        indices_batch += i
        
        if X_unique is None:
            X_unique, indices = X_batch, indices_batch
        else:
            xh_unique_tmp = torch.cat([X_unique, X_batch])
            indices = torch.cat([indices, indices_batch])
            
            X_unique, inverse = torch.unique(xh_unique_tmp, return_inverse=True, dim=0)
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            indices_nonrep = inverse.new_empty(X_unique.size(0)).scatter_(0, inverse, perm)
            indices = indices[indices_nonrep]

    print(f"Filtered {X.shape[0]} points to {X_unique.shape[0]}")
    return X_unique, indices