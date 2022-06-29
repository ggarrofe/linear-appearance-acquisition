from scipy.spatial.transform import Rotation as R
import json
import numpy as np
import os
import sys
import shutil

from tqdm import tqdm

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
                
                pose = np.array(transform_matrix[0])
                if poses is None:
                    poses = pose[None, ...]
                else:
                    poses = np.concatenate((poses, pose[None, ...]))

                # Invert process in https://github.com/NVlabs/instant-ngp/blob/de507662d4b3398163e426fd426d48ff8f2895f6/scripts/colmap2nerf.py#L260
                '''pose[2,:] *= -1 # flip whole world upside down
                c2w=pose[[1,0,2,3],:] # swap y and z'''
                c2w = pose
                c2w[0:3,2] *= -1 # flip the y and z axis
                c2w[0:3,1] *= -1
                w2c = np.linalg.inv(c2w)
                

                # Invert process in https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py
                # must switch to [-u, r, -t] from [r, -u, t]
                # pose[2,:] *= -1
                # c2w = pose[[1,0,2,3],:] 
                # w2c = np.linalg.inv(c2w)
                
                rot = w2c[:3,:3]
                t = w2c[:3, 3]

                r = R.from_matrix(rot)
                quat = r.as_quat()
                #quat = -quat

                line_parts[1] = str(quat[0])
                line_parts[2] = str(quat[1])
                line_parts[3] = str(quat[2])
                line_parts[4] = str(quat[3])
                line_parts[5] = str(t[0])
                line_parts[6] = str(t[1])
                line_parts[7] = str(t[2])

                out_file.write(" ".join(line_parts))

    in_file.close()


if __name__ == "__main__":
    """
    Usage: python3 sparse_known_poses.py <workspace path> transforms.json
            e.g. python3 sparse_known_poses.py ./lego_known_poses transforms.json
    """
    prepare_images(workspace=sys.argv[1], transforms_file=sys.argv[2])
