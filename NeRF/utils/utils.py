

def append_dict(dict, new_dict):
    for key in dict:
        dict[key].append(new_dict[key])
    return dict

def has_nan(tensor):
    