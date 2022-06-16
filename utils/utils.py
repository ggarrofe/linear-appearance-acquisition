from functools import reduce
import numpy as np

def append_dict(dict, new_dict):
    for key in dict:
        dict[key].append(new_dict[key])
    return dict

def has_nan(tensor):
    return tensor.isnan().any()

def summarize_diff(old_arr, new_arr):
    assert(old_arr.shape == new_arr.shape, "The 2 arrays have different shape, cannot find the differences.")
    
    total_items = reduce(lambda x, y: x*y, list(new_arr.shape))
    diff_items = np.where(old_arr != new_arr)
    num_diff = np.count_nonzero(old_arr != new_arr)

    print(f"Changed {num_diff} items out of {total_items} in array of shape {new_arr.shape}")
    if num_diff > 0:
        print(f"\tFrom ({diff_items[0][0]}, {diff_items[1][0]}, {diff_items[2][0]}) to ({diff_items[0][-1]}, {diff_items[1][-1]}, {diff_items[2][-1]})")