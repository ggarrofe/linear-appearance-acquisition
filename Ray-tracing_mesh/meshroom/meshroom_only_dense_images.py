import json
from os import remove
import numpy as np
import glob

def main(cameras_file_origin, cameras_file_dest, prepare_dense_scene):
    with open(cameras_file_origin) as json_file:
        cameras = json.load(json_file)
    
    len_poses = len(cameras["poses"])
    len_views = len(cameras["views"])
    print(f"{len_poses} poses and {len_views} views")

    view_ids = [path.split("/")[-1].split(".exr")[0] for path in glob.glob(prepare_dense_scene+"*.exr")]

    remove_i = []
    for i, pose in enumerate(cameras["poses"]):
        if pose["poseId"] not in view_ids:
            remove_i.append(i)
        
    remove_i.sort(reverse=True)
    for i in remove_i:
        cameras["poses"].pop(i)

    remove_i = []
    for view in cameras["views"]:
        if view["viewId"] not in view_ids:
            remove_i.append(i)

    remove_i.sort(reverse=True)
    for i in remove_i:
        cameras["views"].pop(i)

    with open(cameras_file_dest, 'w') as json_file:
        json.dump(cameras, json_file, indent=4)

    len_poses = len(cameras["poses"])
    len_views = len(cameras["views"])
    print(f"{len_poses} poses and {len_views} views saved")

def copy_filtered_images(cameras_file_origin, imgs_dest, prepare_dense_scene):
    with open(cameras_file_origin) as json_file:
        cameras = json.load(json_file)
    
    len_poses = len(cameras["poses"])
    len_views = len(cameras["views"])
    print(f"{len_poses} poses and {len_views} views")

    view_ids = [path.split("/")[-1].split(".exr")[0] for path in glob.glob(prepare_dense_scene+"*.exr")]

    for view in cameras["views"]:
        if view["viewId"] in view_ids:

if __name__ == "__main__":
    main("./MeshroomCache/KnownCamera/535843973c2b1c1ce8d1e990ed5ea0d5c0f6fdbe/cameras.sfm", 
         "./MeshroomCache/KnownCamera/535843973c2b1c1ce8d1e990ed5ea0d5c0f6fdbe/filtered_cameras.sfm",
         "./MeshroomCache/PrepareDenseScene/715abc1c36ed2539b2087bb73992e7837299b212/")