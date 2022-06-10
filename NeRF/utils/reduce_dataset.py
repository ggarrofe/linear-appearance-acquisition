import numpy as np
import os
import shutil

basedir="../../data/lego_llff/"
smalldir="../../data/lego_llff_small/"


for subdir in ["train", "test", "val"]:
    if not os.path.exists(os.path.join(smalldir, subdir)):
        print("Creating ", os.path.join(smalldir, subdir))
        os.makedirs(os.path.join(smalldir, subdir))

    poses_arr = np.load(os.path.join(basedir, f"poses_bounds_{subdir}.npy"))
    np.save(os.path.join(smalldir, f"poses_bounds_{subdir}.npy"), poses_arr[:20,:]) 

    imgfiles = [os.path.join(basedir, subdir, f) for f in sorted(os.listdir(os.path.join(basedir, subdir))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    for f in imgfiles[:20]:
        print(f"Copying {f}", end="\r")
        f = f.split('/')[-1]
        shutil.copyfile(os.path.join(basedir, subdir, f), os.path.join(smalldir, subdir, f))

    print("Done")