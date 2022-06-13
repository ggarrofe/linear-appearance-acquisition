import numpy as np
import os
import shutil

basedir="../../data/lego_llff/"
smalldir="../../data/lego_llff_small/"
imgs = [[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23], 
        [0, 8, 16, 24], 
        [0, 8, 16, 24]]

for i_dir, subdir in enumerate(["train", "test", "val"]):
    if not os.path.exists(os.path.join(smalldir, subdir)):
        print("Creating ", os.path.join(smalldir, subdir))
        os.makedirs(os.path.join(smalldir, subdir))

    poses_arr = np.load(os.path.join(basedir, f"poses_bounds_{subdir}.npy"))

    imgfiles = [os.path.join(basedir, subdir, f) for f in sorted(os.listdir(os.path.join(basedir, subdir))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    smallposes = np.zeros((len(imgs[i_dir]), 17))
    i_saved = 0

    for i, f in enumerate(imgfiles):
        f = f.split('/')[-1]
        id = f.split('_')[1].split(".png")[0]
        
        if int(id) in imgs[i_dir]:
            print(f"Copying {f}")
            smallposes[i_saved, :] = poses_arr[i, :]
            i_saved += 1
            
            shutil.copyfile(os.path.join(basedir, subdir, f), os.path.join(smalldir, subdir, f))

    np.save(os.path.join(smalldir, f"poses_bounds_{subdir}.npy"), smallposes)
    print("Done")