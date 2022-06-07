import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import open3d as o3d
    

def plot_losses(training_losses, val_losses, it):
    plt.figure(figsize=(9, 4))
    plt.plot(training_losses, label="Training")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.title(f"Losses - it {it}")
    plt.show()

def plt_first_ray_sampling_pdf(pdf, cdf, j, samples):
    fig = plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.plot(pdf[0].detach().cpu().numpy())
    plt.ylabel("PDF")
    plt.title("First ray's PDF")
    
    ax1 = plt.subplot(132)
    ax1.hist(j[0,:].detach().cpu().numpy(), bins=cdf.shape[-1], color='orange')
    ax1.set_ylabel("Counts")

    ax2 = ax1.twinx()
    ax2.plot(cdf[0].detach().cpu().numpy(), color='red')
    ax2.set_ylabel("CDF")

    plt.title("First ray's CDF")
    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['orange','red']]
    labels= ["Selected indices", "CDF"]
    plt.legend(handles, labels)

    samples = samples[0].detach().cpu().numpy()
    
    plt.subplot(133)
    plt.hist(samples[:], bins=cdf.shape[-1])
    plt.title("Selected locations")
    fig.tight_layout()
    plt.show()

def plot_mesh(mesh, mesh_path=None):
    if mesh_path is not None:
        mesh = o3d.io.read_triangle_mesh(mesh_path)

    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])