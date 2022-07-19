from fileinput import filename
import numpy as np
import os
import shutil


def reduce_dataset(basedir="../data/lego_llff/", smalldir="../data/lego_llff_small_2/"):

    imgs = [[1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23],  # train imgs
            [0, 8, 16, 24],  # test imgs
            [0, 8, 16, 24]]  # val imgs

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
                print(f"Copying {f} - {i}")
                smallposes[i_saved, :] = poses_arr[i, :]
                i_saved += 1
                
                shutil.copyfile(os.path.join(basedir, subdir, f), os.path.join(smalldir, subdir, f))

        np.save(os.path.join(smalldir, f"poses_bounds_{subdir}.npy"), smallposes)
        print("Done")

def split_dataset(basedir="../COLMAP/lego_llff", destdir="../data/lego_llff"):
    subdirs = ["train", "val", "test"]
    images = [list(range(0, 100)), list(range(100, 200)), list(range(200, 400))]

    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    imgfiles = [os.path.join(basedir, "images", f) for f in sorted(os.listdir(os.path.join(basedir, "images"))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    for i_dir, subdir in enumerate(subdirs):
        if not os.path.exists(os.path.join(destdir, subdir)):
            print("Creating ", os.path.join(destdir, subdir))
            os.makedirs(os.path.join(destdir, subdir))

        smallposes = np.zeros((len(images[i_dir]), 17))
        i_saved = 0

        for i, f in enumerate(imgfiles):
            f = f.split('/')[-1]
            id = f.split('_')[1].split(".png")[0]
            
            if int(id) in images[i_dir]:
                print(f"Copying {f} - {i} to {destdir}/{subdir}")
                smallposes[i_saved, :] = poses_arr[i, :]
                i_saved += 1
                
                shutil.copyfile(os.path.join(basedir, "images", f), os.path.join(destdir, subdir, f))

        np.save(os.path.join(destdir, f"poses_bounds_{subdir}.npy"), smallposes)
    print("Done")

def clean_dataset(basedir="../COLMAP/lego_llff"):
    subdirs = ["train", "val", "test"]
    images = [[], [], []] #imgs to be removed

    
    for i_dir, subdir in enumerate(subdirs):
        poses_arr = np.load(os.path.join(basedir, f"poses_bounds_{subdir}.npy"))

        imgfiles = [os.path.join(basedir, subdir, f) for f in sorted(os.listdir(os.path.join(basedir, subdir))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        smallposes = np.zeros((len(imgfiles)-len(images[i_dir]), 17))
        i_saved = 0

        for i, f in enumerate(imgfiles):
            f = f.split('/')[-1]
            id = f.split('_')[1].split(".png")[0]
            
            if int(id) in images[i_dir]:
                os.remove(os.path.join(basedir, subdir, f))
                continue

            print(f"Keeping {f} - {i}")
            smallposes[i_saved, :] = poses_arr[i, :]
            i_saved += 1
            
        np.save(os.path.join(basedir, f"poses_bounds_{subdir}.npy"), smallposes)

def convert_nerfactor(basedir="../data/lego_llff", nerfactor_dir="../data/lego_nerfactor", destdir="../data/lego_lighting", light_id=None):
    import PIL.Image
    subdirs = ["train", "val", "test"]
    if light_id is None:
        light_ids = ["0000-0000", "0000-0008", "0000-0016", "0000-0024", "0004-0000", "0004-0008", "0004-0016", "0004-0024"] 
    else:
        light_ids = [light_id]

    for i_dir, subdir in enumerate(subdirs):
        poses_arr = np.load(os.path.join(basedir, f"poses_bounds_{subdir}.npy"))
        imgfiles = [os.path.join(basedir, subdir, f) for f in sorted(os.listdir(os.path.join(basedir, subdir))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        
        newposes = None

        if not os.path.exists(os.path.join(destdir, subdir)):
            print("Creating ", os.path.join(destdir, subdir))
            os.makedirs(os.path.join(destdir, subdir))

        for i, f in enumerate(imgfiles):
            f = f.split('/')[-1]
            id = int(f.split('_')[1].split(".png")[0])

            for light in light_ids:
                origin_path = os.path.join(nerfactor_dir, f"{subdir}_{id:03}", f"rgba_olat-{light}.png")
                dest_path = os.path.join(destdir, subdir, f"{id}_rgba_olat-{light}.png")

                if not os.path.exists(origin_path):
                    continue

                print(f"Copying {origin_path} - {i} to {dest_path}")

                if newposes is None:
                    newposes = poses_arr[None, i, ...]
                else:
                    newposes = np.concatenate((newposes, poses_arr[None, i, ...]))


                rgba_image = PIL.Image.open(origin_path)
                rgb_image = rgba_image.convert('RGB')
                rgb_image.save(dest_path)
                #shutil.copyfile(origin_path, dest_path)

def add_to_id(basedir="../data/lego_lighting/val", offset=300):
    for f in os.listdir(os.path.join(basedir)):
        print(f)
        parts = f.split("_")
        id = int(parts[0])
        name = '_'.join(parts[1:])
        id += offset
        filename = f"{str(id)}_{name}"
        print(f"\t{filename}")
        shutil.copyfile(f"{basedir}/{f}", f"../COLMAP/lego_lighting/images/{filename}")


if __name__ == "__main__":
    #convert_nerfactor(light_id="0000-0000")
    add_to_id()