import open3d as o3d
import sys

# usage python3 poisson_mesher.py ./lego_llff/dense/fused.ply ./lego_llff/dense/meshed-poisson.ply
if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(sys.argv[1])

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    o3d.io.write_triangle_mesh(sys.argv[2], mesh)