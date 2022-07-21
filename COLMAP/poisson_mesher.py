import open3d as o3d
import sys
import numpy as np
import matplotlib.pyplot as plt

# usage python3 poisson_mesher.py ./lego_llff/dense/fused.ply ./lego_llff/dense/meshed-poisson.ply
if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(sys.argv[1])

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

    o3d.visualization.draw_geometries([mesh])

    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    o3d.visualization.draw_geometries([density_mesh])

    vertices_to_remove = densities < np.quantile(densities, 0.03)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    o3d.visualization.draw_geometries([mesh])

    
    o3d.io.write_triangle_mesh(sys.argv[2], mesh)