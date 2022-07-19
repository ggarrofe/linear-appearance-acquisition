from xmlrpc.client import Transport
from scipy.spatial.transform import Rotation as R
import json
import numpy as np
import os
import sys
import shutil

from tqdm import tqdm

def get_pose(transform_matrix):
    blender_to_colmap_rotation = np.diag([1,-1,-1])
    # Convert from blender world space to view space
    blender_world_translation = transform_matrix[:3, 3]
    blender_world_rotation = transform_matrix[:3, :3]
    blender_view_rotation = blender_world_rotation.T
    blender_view_translation = -1.0 * blender_view_rotation @ blender_world_translation
    # Convert from blender view space to colmap view space
    colmap_view_rotation = blender_to_colmap_rotation @ blender_view_rotation
    colmap_view_rotation_quaternion = R.from_matrix(colmap_view_rotation).as_quat()
    colmap_view_translation = blender_to_colmap_rotation @ blender_view_translation
    return [
        str(colmap_view_rotation_quaternion[3]),
        str(colmap_view_rotation_quaternion[0]),
        str(colmap_view_rotation_quaternion[1]),
        str(colmap_view_rotation_quaternion[2]),
        str(colmap_view_translation[0]),
        str(colmap_view_translation[1]),
        str(colmap_view_translation[2])
    ]

def prepare_images(workspace, transforms_file="transforms.json", i_model=0):
    workspace = workspace if workspace[-1] != '/' else workspace[:-1]

    if not os.path.exists(workspace+"/sparse_known_poses_manual"):
        os.makedirs(workspace+"/sparse_known_poses_manual")

    if not os.path.exists(workspace+"/sparse_known_poses_triangulated"):
        os.makedirs(workspace+"/sparse_known_poses_triangulated")

    file = open(workspace+'/sparse_known_poses_manual/points3D.txt', 'w')
    file.close()
    shutil.copy(workspace+f'/sparse/{i_model}/cameras.txt', workspace+'/sparse_known_poses_manual/cameras.txt')

    in_file = open(workspace+f'/sparse/{i_model}/images.txt', 'r')
    with open(workspace+'/sparse_known_poses_manual/images.txt', 'w') as out_file:
        with open(transforms_file) as json_file:
            transforms = json.load(json_file)
            poses = None
            for i, line in tqdm(enumerate(in_file), unit="matrix", desc="Computing world 2 camera matrices"):
                if i < 5:
                    out_file.write(line)
                    continue
                elif i % 2 == 1:
                    out_file.write("\n")
                    continue

                line_parts = line.split(" ")

                file = line_parts[-1].split(".png")[0]
                transform_matrix = [transforms["frames"][j]["transform_matrix"] 
                                    for j in range(len(transforms["frames"])) 
                                        if transforms["frames"][j]["file_path"].split("/")[-1] == file]
                assert len(transform_matrix)>0, f"No transform matrix found for {file}"
                
                transform_matrix = np.array(transform_matrix[0])
                print(transform_matrix.shape)
                #pose = get_pose(transform_matrix)
                #print(pose)

                pose_json = [transforms["frames"][j]["COLMAP_transform_matrix"] 
                                    for j in range(len(transforms["frames"])) 
                                        if transforms["frames"][j]["file_path"].split("/")[-1] == file]
                assert len(transform_matrix)>0, f"No COLMAP transform matrix found for {file}"
                pose_json = pose_json[0]

                #print(pose_json)
                line_parts[1]=str(pose_json['w_rotation'])
                line_parts[2]=str(pose_json['x_rotation'])
                line_parts[3]=str(pose_json['y_rotation'])
                line_parts[4]=str(pose_json['z_rotation'])
                line_parts[5]=str(pose_json['x_pos'])
                line_parts[6]=str(pose_json['y_pos'])
                line_parts[7]=str(pose_json['z_pos'])
                out_file.write(" ".join(line_parts))

    in_file.close()


if __name__ == "__main__":
    """
    Usage: python3 sparse_known_poses.py <workspace path> transforms.json
            e.g. python3 sparse_known_poses.py ./lego_known_poses transforms.json
    """
    prepare_images(workspace=sys.argv[1], transforms_file=sys.argv[2])
