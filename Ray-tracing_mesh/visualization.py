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

def create_lineset(origins, dirs, color, near=2, far=6):
    points_n = origins[::3300] + near * dirs[::3300]
    points_f = origins[::3300] + far * dirs[::3300]
    points = torch.concat([points_n, points_f])

    lines = [(i, i+points_n.shape[0]) for i in range(points_n.shape[0])]
    colors = [color for i in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set

def plot_rays_and_mesh(rays_od, mesh, light_rays=None, xh=None, near=2, far=6, pose=None):
    mesh.compute_vertex_normals()

    camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    camera.translate(rays_od[0, :3])
    camera.paint_uniform_color([1, 0, 0])
    
    camera_rays = create_lineset(rays_od[..., :3], rays_od[..., 3:], color=np.array([1, 0, 0]), near=near, far=500) # red camera rays
    print("dev", rays_od.device, pose.device, rays_od.type(), rays_od.shape)
    up_dir = pose[:3,:3].float() @ torch.tensor([0, 1., 0])
    opengl2opencv = torch.tensor([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]]).float()
    origin = pose[:3, 3:4].T.float()
    print("origin", origin)
    print("up", up_dir)
    up_vector = create_lineset(origin, up_dir[None, ...], color=np.array([1, 0, 1]), near=0, far=50)
    geometries = [camera, mesh, camera_rays, up_vector]

    if pose is not None:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=pose[:3,3])
        mesh_frame.rotate(pose[:3,:3].numpy())
        geometries.append(mesh_frame)

    if light_rays is not None:
        light = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        light.translate(light_rays[0, :3])
        light.paint_uniform_color([1, 0.706, 0])
        geometries.append(light)
        
        light_rays = create_lineset(light_rays[..., :3], light_rays[..., 3:], color=np.array([1, 0.706, 0]), near=0, far=far) # yellow light rays
        geometries.append(light_rays)

    if xh is not None:
        H = create_lineset(origins=xh[..., :3], dirs=xh[..., 3:], color=np.array([0, 0, 1]), near=0, far=4) # yellow light rays
        geometries.extend([H])
        
        surf_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        surf_point.translate(xh[0, :3])
        surf_point.paint_uniform_color([0, 0, 1])
        geometries.append(surf_point)

    
    o3d.visualization.draw_geometries(geometries)

def plot_camera_and_mesh(mesh_path, camera_pose):
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.io.read_triangle_mesh(mesh_path, print_progress=True)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    mesh.compute_vertex_normals()

    camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    camera.translate(camera_pose[:3, 3])
    camera.paint_uniform_color([1, 0, 0])
    
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=camera_pose[:3,3])
    cam_frame.rotate(camera_pose[:3,:3].numpy())
    geometries.append(cam_frame)
    
    geometries = [camera, mesh, cam_frame]
    
    o3d.visualization.draw_geometries(geometries)

def render_mesh(mesh_path, camera_pose):
    # To install dependencies: pip install numpy, open3d, opencv-python

    # Create a renderer with the desired image size
    img_width = 1000
    img_height = 1000
    render = o3d.visualization.rendering.OffscreenRenderer(img_width, img_height)

    # Pick a background colour (default is light gray)
    #render.scene.set_background([0.1, 0.2, 0.3, 1.0])  # RGBA

    # Create the mesh geometry.
    # (We use arrows instead of a sphere, to show the rotation more clearly.)
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"

    mesh = o3d.io.read_triangle_mesh(mesh_path, print_progress=True)
    mesh.compute_vertex_normals()
    render.scene.add_geometry("model", mesh, mtl)

    # Create a copy of the above mesh and rotate it around the origin.
    # (If the original mesh is not going to be used, we can just rotate it directly without making a copy.)
    '''R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    mesh_r = copy.deepcopy(mesh)
    mesh_r.rotate(R, center=(0, 0, 0))'''

    # Show the original coordinate axes for comparison.
    # X is red, Y is green and Z is blue.
    render.scene.show_axes(True)

    # Define a simple unlit Material.
    # (The base color does not replace the arrows' own colors.)
    '''mtl = o3d.visualization.rendering.Material()
    mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
    mtl.shader = "defaultUnlit"'''

    # Add the arrow mesh to the scene.
    # (These are thicker than the main axis arrows, but the same length.)
    '''render.scene.add_geometry("rotated_model", mesh_r, mtl)'''

    # Since the arrow material is unlit, it is not necessary to change the scene lighting.
    render.scene.scene.enable_sun_light(True)
    render.scene.set_lighting(render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
    #render.scene.add_point_light("pointlight", np.array([1, 1, 1]), camera_pose[:3, 3].numpy(), 50, 8,False)

    # Optionally set the camera field of view (to zoom in a bit)
    vertical_field_of_view = 15.0  # between 5 and 90 degrees
    #vertical_field_of_view = 54.0
    aspect_ratio = img_width / img_height  # azimuth over elevation
    near_plane = 0.1
    far_plane = 50.0
    fov_type = o3d.visualization.rendering.Camera.FovType.Vertical
    #render.scene.camera.set_projection(vertical_field_of_view, aspect_ratio, near_plane, far_plane, fov_type)

    # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
    opengl2opencv = np.diag([1,-1,-1])

    #origin = pose[:3, 3:4].T.float()

    center = np.array([0, 0, 0])  # look_at target
    eye = camera_pose[:3, 3:4].numpy() #[0, 0, 10]  # camera position
    up = camera_pose[:3,:3].float().numpy() @ np.array([0, 1, 0]) # camera orientation
    #render.scene.camera.look_at(center, eye, up)

    camK = np.array([[886.81,0.0,512.0],[0.0,886.81,512.0],[0.0,0.0,1.0]], dtype="float64")
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic
    print(type(camera_intrinsic))
    print(type(1024), type(1024), type(float(camK[0,0])),type(float(camK[1,1])), type(float(camK[0,2])),type(float(camK[1,2])))
    camera_intrinsic.set_intrinsics(1024, 1024, float(camK[0,0]),float(camK[1,1]), 
                                               float(camK[0,2]),float(camK[1,2]))
    render.setup_camera(camera_intrinsic, camera_pose)

    # Read the image into a variable
    img_o3d = render.render_to_image()

    plt.imshow(np.array(img_o3d))
    plt.show()
    # Display the image in a separate window
    # (Note: OpenCV expects the color in BGR format, so swop red and blue.)
    '''img_cv2 = cv2.cvtColor(np.array(img_o3d), cv2.COLOR_RGBA2BGRA)
    cv2.imshow("Preview window", img_cv2)
    cv2.waitKey()'''

    # Optionally write it to a PNG file
    o3d.io.write_image("output.png", img_o3d, 9)

def render_mesh_2(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path, print_progress=True)
    mesh.compute_vertex_normals()
	
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False) #works for me with False, on some systems needs to be true
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("output.png")
    vis.destroy_window()


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