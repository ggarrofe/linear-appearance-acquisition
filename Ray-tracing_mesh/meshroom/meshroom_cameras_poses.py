import json
from os import remove
import numpy as np
from PySide2.QtGui import QMatrix4x4, QMatrix3x3, QQuaternion, QVector3D, QVector2D

def main(transforms_file, cameras_file_origin, cameras_file_dest):
    with open(transforms_file) as json_file:
        transforms = json.load(json_file)
    with open(cameras_file_origin) as json_file:
        cameras = json.load(json_file)
    
    len_poses = len(cameras["poses"])
    len_views = len(cameras["views"])
    print(f"{len_poses} poses and {len_views} views")

    for i, view in enumerate(cameras["views"]):
        print(f"View {i}/{len_views}", end="\r")
        pose_id = view["poseId"]
        path = view["path"]
        file = path.split("/")[-1].split(".png")[0]
        
        i_pose = [j for j in range(len(cameras["poses"])) if cameras["poses"][j]["poseId"] == pose_id]
        
        transform_matrix = [transforms["frames"][j]["transform_matrix"] for j in range(len(transforms["frames"])) if transforms["frames"][j]["file_path"].split("/")[-1] == file]
        #transform_matrix = np.array(transform_matrix)
        
        '''rot = transform_matrix[:, :3, :3].flatten()
        rot = np.array([[rot[0], -rot[1], -rot[2]],
               [rot[3], -rot[4], -rot[5]],
               [rot[6], -rot[7], -rot[8]]])'''

        #rot = np.squeeze(transform_matrix[:, :3, :3]).T

        c2w = np.array(transform_matrix)

        # Invert process in https://github.com/NVlabs/instant-ngp/blob/de507662d4b3398163e426fd426d48ff8f2895f6/scripts/colmap2nerf.py#L260
        c2w[2,:] *= -1 # flip whole world upside down
        c2w=c2w[[1,0,2,3],:] # swap y and z
        c2w[0:3,2] *= -1 # flip the y and z axis
        c2w[0:3,1] *= -1
        m = np.linalg.inv(c2w)
        rot = m[:3,:3]
        t = m[:3, 3]
        
        center = t

        pose = {}
        pose["poseId"] = pose_id
        pose["pose"] = {}
        pose["pose"]["transform"] = {}
        pose["pose"]["transform"]["rotation"] = [f'{a:.18f}' for a in rot.flatten()]
        # center: R.transpose()*(-translate)
        center = np.squeeze(transform_matrix[:, :3, 3])
        pose["pose"]["transform"]["center"] = [f'{a:.18f}' for a in center.flatten()]
        pose["pose"]["locked"] = "1"

        if len(i_pose) != 0:
            cameras["poses"][i_pose[0]] = pose
        else:
            cameras["poses"].append(pose)

    cameras.pop('featuresFolders', None)
    cameras.pop('matchesFolders', None)

    len_poses = len(cameras["poses"])
    len_views = len(cameras["views"])

    with open(cameras_file_dest, 'w') as json_file:
        json.dump(cameras, json_file, indent=4)

    print(f"{len_poses} poses and {len_views} views saved")

if __name__ == "__main__":
    main("transforms_train.json", "./MeshroomCache/StructureFromMotion/535843973c2b1c1ce8d1e990ed5ea0d5c0f6fdbe/cameras.sfm", "./MeshroomCache/KnownCamera/535843973c2b1c1ce8d1e990ed5ea0d5c0f6fdbe/cameras.sfm")