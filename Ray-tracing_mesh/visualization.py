import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

import open3d as o3d
import open3d.core as o3c
import copy
import numpy as np

def plot_mesh(mesh, mesh_path=None):
    if mesh_path is not None:
        mesh = o3d.io.read_triangle_mesh(mesh_path)

    mesh.compute_vertex_normals()
    mesh_frame = o3d.geometry.TriangleMesh.create_mesh_coordinate_frame(size=1.0, origin=[0., 0., 0.])
    o3d.visualization.draw_geometries([mesh, mesh_frame])

def plot_rays_and_mesh(rays_od, mesh, hwf, near=2, far=6, rgb=None, rot_matrix=None, light_source=None):
    points_n = rays_od[::100, :3] + near * rays_od[::100, 3:]
    points_f = rays_od[::100, :3] + far * rays_od[::100, 3:]
    points = torch.concat([points_n, points_f])

    lines = [(i, i+points_n.shape[0]) for i in range(points_n.shape[0])]
    if rgb is not None:
        rgb = list(rgb)
    else:
        rgb = [1, 0, 0]
    colors = [rgb for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    camera.translate(rays_od[0, :3])

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([line_set, camera])
    return
    geometries = [line_set, camera, mesh]

    if rot_matrix is not None:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
        coord_frame_rot = copy.deepcopy(coord_frame)
        coord_frame_rot.rotate(rot_matrix[:3,:3].numpy())
        geometries.append(coord_frame_rot)

        print(rot_matrix[:, 3:])

    if light_source is not None:
        light = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        light.paint_uniform_color([1, 0.706, 0])
        light_source = rot_matrix[:3,:3] @ light_source
        light_source += rot_matrix[:, 3:]
        light.translate(light_source, relative=False)
        #geometries.append(light)

    o3d.visualization.draw_geometries(geometries)


def get_lines_camera(rays_od, rgb, near=2, far=6):
    points_n = rays_od[::100, :3] + near * rays_od[::100, 3:]
    points_f = rays_od[::100, :3] + far * rays_od[::100, 3:]
    points = torch.concat([points_n, points_f])

    lines = [(i, i+points_n.shape[0]) for i in range(points_n.shape[0])]
    if rgb is not None:
        rgb = list(rgb)
    else:
        rgb = [1, 0, 0]
    colors = [rgb for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    camera.paint_uniform_color(rgb)
    camera.translate(rays_od[0, :3])

    return line_set, camera


def plot_test_rays(rays_nerf, rays_llff, rays_colmap, mesh, hwf, near=2, far=6, rot_matrix=None):
    line_set_nerf, camera_nerf = get_lines_camera(rays_nerf, rgb=[1, 0, 0], near=near, far=far)
    line_set_llff, camera_llff = get_lines_camera(rays_llff, rgb=[0,1,0], near=near, far=far)
    line_set_colmap, camera_colmap = get_lines_camera(rays_colmap, rgb=[0,0,1], near=near, far=far)

    
    coord_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0., 0., 0.])
    coord_system_2 = copy.deepcopy(coord_system)
    coord_system_2.rotate(rot_matrix[:3,:3].numpy())

    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([line_set_nerf, camera_nerf, 
                                       line_set_llff, camera_llff, 
                                       #line_set_colmap, camera_colmap,
                                       mesh,  coord_system_2])#coord_system,