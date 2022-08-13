import numpy as np
import os
import shutil
import PIL.Image
import json

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

def split_dataset(basedir="../COLMAP/lego_llff", destdir="../data/lego"):
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

def get_subdirs(path) -> list:
    """ Check if the provided path contains the images or has other inner directories that 
        contain the images. Ommits the resized folders that end with a number.

        e.g.: dataset           Has the following subdirs: ["train", "test", "val"]
                |-> train
                |-> test
                |-> val
                |-> train_1

                dataset           Does not have subdirs.
                |-> img0.png
                |-> img1.png
                ...

                dataset           Does not have subdirs and will use the images in dataset/
                |-> docs
                |-> img0.png
                ...

    Args:
        path (str): Data path

    Returns:
        list: list of subdirs, if there are no subdirs will return a list with ['.']
    """
    subdirs = []

    if path[-1] != '/':
        path = path+'/'

    if not any(fname.endswith('.png') for fname in os.listdir(path)):
        subdirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and not name[-1].isdigit() and not ".tmp" in name]

    if len(subdirs) == 0:
        subdirs = ['.'] 
    
    return subdirs

def convert_nerfactor(basedir="../data/lego_llff", nerfactor_dir="../data/lego_nerfactor", destdir="../data/lego_lighting", light_id=None):
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

def merge_images(sourcedir="../data/hotdog", destdir="../COLMAP/hotdog/images", is_synthetic=True):
    subdirs = ["train", "val"]
    i_img = 0
    transforms = None
    for subdir in subdirs:

        if is_synthetic:
            with open(os.path.join(sourcedir, f"transforms_{subdir}.json")) as json_file:
                if transforms is None:
                    transforms = json.load(json_file)
                else:
                    data = json.load(json_file)
                    transforms['frames'].extend(data['frames'])

        for i, f in enumerate(sorted(os.listdir(os.path.join(sourcedir, subdir)))):
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'):
                source_path = os.path.join(sourcedir, subdir, f)
                dest_path = os.path.join(destdir, f"r_{i_img}.png")
                print(f"Copying {source_path} - {i} to {dest_path}")

                rgba_image = PIL.Image.open(source_path)
                rgb_image = rgba_image.convert('RGB')
                rgb_image.save(dest_path)

                if is_synthetic:
                    path = '/'.join(transforms['frames'][i_img]['file_path'].split('/')[:-1])
                    transforms['frames'][i_img]['file_path'] = f'{path}/r_{i_img}'
                i_img += 1

    if is_synthetic:
        with open(os.path.join(destdir, "..", f"transforms.json"), "w") as json_file:
            json.dump(transforms, json_file, indent=4)

def merge_images_realitycapture(sourcedir="../data/lego", destdir="../RealityCapture/lego"):
    subdirs = ["train", "val"]
    i_img = 0
    transforms = None
    for subdir in subdirs:

        with open(os.path.join(sourcedir, f"transforms_{subdir}.json")) as json_file:
            transforms = json.load(json_file)

        for i, f in enumerate(sorted(os.listdir(os.path.join(sourcedir, subdir)))):
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png'):
                source_path = os.path.join(sourcedir, subdir, f)
                dest_path = os.path.join(destdir, f"r_{i_img}.png")
                file_path = os.path.join(destdir, f"r_{i_img}.txt")
                print(f"Copying {source_path} - {i} to {dest_path}")

                rgba_image = PIL.Image.open(source_path)
                rgb_image = rgba_image.convert('RGB')
                rgb_image.save(dest_path)

                transform = np.array(transforms['frames'][i]['transform_matrix'][:3])
                with open(file_path, "w") as transform_file:
                    for i in range(len(transform)):
                        row = [str(c) for c in transform[i]]
                        transform_file.write(' '.join(row)+'\n')
                        
                i_img += 1

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def realitycapture2blender_pose(transform_matrix):
    # https://support.capturingreality.com/hc/en-us/articles/360017783459-RealityCapture-XMP-Camera-Math
    #realiycapture2nerf_rotation = np.diag([1,-1,-1])
    realitycapture2blender_rotation = np.array([[1, 0, 0],
                                             [0, -1, 0],
                                             [0, 0, -1]])
    # Convert from blender world space to view space
    blender_world_translation = transform_matrix[:3, 3]
    blender_world_scale = np.linalg.norm(transform_matrix[:3, :3], axis=0)
    blender_world_rotation = transform_matrix[:3, :3]
    blender_world_rotation /= blender_world_scale

    blender_view_rotation = blender_world_rotation.T
    #blender_view_translation = -1.0 * blender_view_rotation @ blender_world_translation
    # Convert from blender view space to colmap view space
    realitycapture_view_rotation = blender_view_rotation @ realitycapture2blender_rotation
    #realitycapture_view_translation = blender_view_translation @ realitycapture2nerf_rotation
    
    trans_mat = np.concatenate([realitycapture_view_rotation, blender_world_translation.reshape(3,1)], axis=1)
    trans_mat = np.concatenate([trans_mat, np.array([[0, 0, 0, 1]])], axis=0)
    return trans_mat

def blender2realitycapture_pose(transform_matrix):
    realitycapture2blender_rotation = np.diag([1,-1,-1])

    # Convert from realitycapture world space to view space
    blender_world_translation = transform_matrix[:3, 3]
    realitycapture_view_rotation = transform_matrix[:3, :3]

    blender_view_rotation = realitycapture_view_rotation @ realitycapture2blender_rotation
    blender_world_rotation = blender_view_rotation.T
    
    trans_mat = np.concatenate([blender_world_rotation, blender_world_translation.reshape(3,1)], axis=1)
    trans_mat = np.concatenate([trans_mat, np.array([[0, 0, 0, 1]])], axis=0)
    return trans_mat

def xmp2transforms(sourcedir, xmp_path, realitycapture_path):
    subdirs = ["train", "val"]
    indices = [list(range(0, 100)), list(range(100, 200))]
    
    transforms = []
    for i, subdir in enumerate(subdirs):        
        transforms.append({
            "camera_angle_x": 0,
            "frames": []
        })

        if not os.path.exists(os.path.join(realitycapture_path, subdir)):
            print("Creating ", os.path.join(realitycapture_path, subdir))
            os.makedirs(os.path.join(realitycapture_path, subdir))
        
    for f in sorted(os.listdir(xmp_path)):
        
        frame = {
            "file_path": f.rsplit('.xmp', 1)[0],
        }

        id = int(frame["file_path"].split("_")[1])
        subdir = None
        i_subdir = -1
        for i, subd in enumerate(subdirs):
            if id in indices[i]: 
                subdir = subd
                i_subdir = i
                break

        print(f"Getting the transforms of {frame['file_path']}")
        with open(f"{xmp_path}/{f}", "r") as xmp_file:
            rot_mat = None
            t = None
            for line in xmp_file:
                if '<xcr:Rotation>' in line:
                    line = line.strip()
                    line = line.split('<xcr:Rotation>', 1)[1]
                    line = line.rsplit('</xcr:Rotation>', 1)[0]
                    rot_mat = np.array([float(i) for i in line.split(" ")]).reshape((3,3))
                
                if '<xcr:Position>' in line:
                    line = line.strip()
                    line = line.split('<xcr:Position>', 1)[1]
                    line = line.rsplit('</xcr:Position>', 1)[0]
                    t = np.array([float(i) for i in line.split(" ")]).reshape((3,1))

                if 'xcr:Position=' in line:
                    line = line.strip()
                    line = line.split('xcr:Position="', 1)[1]
                    line = line.rsplit('"', 1)[0]
                    t = np.array([float(i) for i in line.split(" ")]).reshape((3,1))

                if 'xcr:FocalLength35mm=' in line:
                    line = line.strip()
                    line = line.split('xcr:FocalLength35mm="', 1)[1]
                    line = line.rsplit('"', 1)[0]
                    focal = float(line)
                    cam_ang_x = 2 * np.arctan(36/(2*focal))

            trans_mat = np.concatenate([rot_mat, t], axis=1)
            trans_mat = np.concatenate([trans_mat, np.array([[0, 0, 0, 1]])], axis=0)
            #print("pose realitycapt", trans_mat)
            pose = realitycapture2blender_pose(trans_mat)
            #print("pose synth", pose)
            #print("pose reconstr", blender2realitycapture_pose(pose))
            #pose = np.concatenate([pose[:, 0:1], -pose[:, 1:2], -pose[:, 2:3], pose[:, 3:]], 1)
            
            frame["transform_matrix"] = listify_matrix(pose)
            transforms[i_subdir]["frames"].append(frame)
            transforms[i_subdir]["camera_angle_x"] += 1/len(transforms[i_subdir]["frames"]) * (cam_ang_x - transforms[i_subdir]["camera_angle_x"])
            print("cam angle x", transforms[i_subdir]["camera_angle_x"])

        source_path = os.path.join(sourcedir, f"r_{id}.png")
        dest_path = os.path.join(realitycapture_path, subdir, f"r_{id}.png")
        print(f"Copying {source_path} - {i} to {dest_path}")

        rgba_image = PIL.Image.open(source_path)
        rgb_image = rgba_image.convert('RGB')
        rgb_image.save(dest_path)
            

    for i, subdir in enumerate(subdirs):
        with open(os.path.join(realitycapture_path, f"transforms_{subdir}.json"), "w") as json_file:
            json.dump(transforms[i], json_file, indent=4)

def transforms2xmp(basedir, xmp_path, dest_path):
    subdirs = ["train", "val"]
    indices = [list(range(0, 100)), list(range(100, 200))]

    transforms = []
    for i_subdir, subdir in enumerate(subdirs):
        with open(os.path.join(basedir, f"transforms_{subdir}.json")) as json_file:
            transforms.append(json.load(json_file)) 

    for f in sorted(os.listdir(xmp_path)):
        id = int(f.rsplit('.xmp', 1)[0].split("_")[1])
        subdir = None
        i_subdir = -1
        for i, subd in enumerate(subdirs):
            if id in indices[i]: 
                subdir = subd
                i_subdir = i
                frame_i = id - indices[i_subdir][0]
                break

        source_path = os.path.join(basedir, subdir, f"r_{frame_i}.png")
        dest_img_path = os.path.join(dest_path, f"r_{id}.png")
        print(f"Copying {source_path} - {i} to {dest_img_path}")

        rgba_image = PIL.Image.open(source_path)
        rgb_image = rgba_image.convert('RGB')
        rgb_image.save(dest_img_path)
        
        transform_mat = np.array(transforms[i_subdir]["frames"][frame_i]["transform_matrix"])
        transform_mat = blender2realitycapture_pose(transform_mat)
        cam_ang_x = float(transforms[i_subdir]["camera_angle_x"])
        print("cam ang", cam_ang_x)
        focal_length = 36/(2 * np.tan(cam_ang_x/2))
        
        rotation = list(transform_mat[:3, :3].flatten())
        position = list(transform_mat[:3, 3].flatten())

        with open(f"{xmp_path}/{f}", "r") as xmp_r_file:
            with open(f"{dest_path}/{f}", "w") as xmp_w_file:
                for line in xmp_r_file:
                    if '<xcr:Rotation>' in line:
                        new_line = line.split('<xcr:Rotation>', 1)[0]
                        new_line += '<xcr:Rotation>'
                        new_line += ' '.join([str(r) for r in rotation])
                        new_line += '</xcr:Rotation>'
                        new_line += line.rsplit('</xcr:Rotation>', 1)[1]
                        
                    elif '<xcr:Position>' in line:
                        new_line = line.split('<xcr:Position>', 1)[0]
                        new_line += '<xcr:Position>'
                        new_line += ' '.join([str(p) for p in position])
                        new_line += '</xcr:Position>'
                        new_line += line.rsplit('</xcr:Position>', 1)[1]

                    elif 'xcr:Position=' in line:
                        print(f"special position {f}")
                        new_line = line.split('xcr:Position="', 1)[0]
                        new_line += 'xcr:Position="'
                        new_line += ' '.join([str(p) for p in position])
                        new_line += '"'
                        new_line += line.rsplit('"', 1)[1]

                    elif 'xcr:FocalLength35mm=' in line:
                        print("focal length", focal_length)
                        new_line = line.split('xcr:FocalLength35mm="', 1)[0]
                        new_line += 'xcr:FocalLength35mm="'
                        new_line += str(focal_length)
                        new_line += '"'
                        new_line += line.rsplit('"', 1)[1]

                    else:
                        new_line = line

                    xmp_w_file.write(new_line)

            pass
    


if __name__ == "__main__":
    #convert_nerfactor(light_id="0000-0000")
    #add_to_id()
    #merge_images("../data/lego_llff", "../RealityCapture/lego", is_synthetic=False)
    #merge_images_realitycapture()
    #xmp2transforms("../RealityCapture/hotdog", "../RealityCapture/hotdog_xmp/", "../RealityCapture/hotdog_transforms/")
    #transforms2xmp("../data/hotdog", "../RealityCapture/hotdog_xmp", "../RealityCapture/hotdog_known_xmp")
    split_dataset()
