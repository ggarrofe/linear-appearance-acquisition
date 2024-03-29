import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import wandb
import numpy as np
from PIL import Image

import sys
sys.path.append('../')
sys.path.append('drive/Othercomputers/MacBookPro/')
from utils import utils

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

def validation_view(rgb_map, val_target, img_shape, it=0, out_path=None, name="validation"):
    rgb_map = torch.reshape(rgb_map, img_shape)
    val_target = torch.reshape(val_target, img_shape)
    
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(rgb_map.numpy())
    plt.title(f"Reconstruction - it {it}")
    plt.subplot(122)
    plt.imshow(val_target.numpy())
    plt.title("Target image")

    wandb.log({f"{name}_view": fig}, step=it)
    if out_path is not None:
        plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)

def validation_view_rgb_xndv(rgb_map, val_target, img_shape, points, normals, depths, viewdirs, it=0, out_path=None, name="validation_xnv", save=True, wandb_act=True):
    rgb_map = torch.reshape(torch.clamp(rgb_map, min=0.0, max=1.0), img_shape)
    val_target = torch.reshape(val_target, img_shape)
    normals = torch.reshape(torch.clamp(normals, min=0.0, max=1.0), img_shape)
    viewdirs = torch.reshape(torch.clamp(viewdirs, min=0.0, max=1.0), img_shape)

    min_point = -1.3 #torch.min(points)
    max_point = 3.85 #torch.max(points-min_point)
    points = torch.reshape((points-min_point)/max_point, img_shape)
    min_depth = 1000.0
    max_depth = 10000.0
    depths = torch.reshape((depths-min_depth)/max_depth, list(img_shape)[0:2])
    depths = torch.nan_to_num(depths, posinf=0.0, neginf=0.0, nan=0.0)

    #depths /= torch.max(depths)

    fig = plt.figure(figsize=(20, 7))
    plt.subplot(231)
    plt.imshow(rgb_map.cpu().numpy())
    plt.title(f"Reconstruction - it {it}")
    plt.subplot(232)
    plt.imshow(val_target.cpu().numpy())
    plt.title("Target image")
    plt.subplot(233)
    plt.imshow(points.cpu().numpy())
    plt.title("3D points")
    plt.subplot(234)
    plt.imshow(normals.cpu().numpy())
    plt.title("Normals")
    plt.subplot(235)
    plt.imshow(viewdirs.cpu().numpy())
    plt.title("View directions")
    plt.subplot(236)
    plt.imshow(depths.cpu().numpy())
    plt.title("Depths")

    if save and out_path is not None:
        plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)
        im = Image.fromarray((rgb_map.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_pred_it{it}.png")
        im = Image.fromarray((normals.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_normals_it{it}.png")
        im = Image.fromarray((points.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_points_it{it}.png")
        im = Image.fromarray((depths.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_depths_it{it}.png")

    if wandb_act:
        wandb.log({f"{name}_view": fig}, step=it)
        plt.close(fig)
    else:
        plt.show()
    

'''def validation_view_reflectance(reflectance, specular, diffuse, target, img_shape, points, it=0, out_path=None, name="val_reflectance", save=True, wandb_act=True):
    reflectance = torch.reshape(reflectance, img_shape)
    specular = torch.reshape(specular, img_shape)
    diffuse = torch.reshape(diffuse, img_shape)
    target = torch.reshape(target, img_shape)
    points = torch.reshape(points, img_shape)
    #points /= 7.0 # Keep the values lower than 1, constant so that all the views are scaled the same way

    fig = plt.figure(figsize=(25, 10))
    plt.subplot(221)
    plt.imshow(diffuse.cpu().numpy())
    plt.title(f"Diffuse - it {it}")
    plt.subplot(222)
    plt.imshow(specular.cpu().numpy())
    plt.title(f"Specular - it {it}")
    plt.subplot(223)
    plt.imshow(reflectance.cpu().numpy())
    plt.title("Predicted reflectance")
    plt.subplot(224)
    plt.imshow(target.cpu().numpy())
    plt.title("Target reflectance")

    if save and out_path is not None:
        plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)
        plt.close(fig)
        im = Image.fromarray((diffuse.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_diffuse_it{it}.png")
        im = Image.fromarray((specular.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_specular_it{it}.png")
        im = Image.fromarray((reflectance.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_reflectance_it{it}.png")
    else:
        plt.show()
            
    if wandb_act:
        wandb.log({f"{name}_view": fig}, step=it)
'''

def validation_view_reflectance_enh(reflectance, specular, diffuse, target, linear, img_shape, it=0, out_path=None, name="val_reflectance", save=True, wandb_act=True):
    reflectance = torch.reshape(torch.clamp(reflectance, min=0.0, max=1.0), img_shape)
    lin_diffuse = torch.reshape(torch.clamp(diffuse[0].detach().cpu(), min=0.0, max=1.0), img_shape)
    lin_specular = torch.reshape(torch.clamp(specular[0].detach().cpu(), min=0.0, max=1.0), img_shape)
    enh_diffuse = torch.reshape(torch.clamp(diffuse[1].detach().cpu(), min=0.0, max=1.0), img_shape)
    enh_specular = torch.reshape(torch.clamp(specular[1].detach().cpu(), min=0.0, max=1.0), img_shape)
    target = torch.reshape(target, img_shape)
    linear = torch.reshape(torch.clamp(linear, min=0.0, max=1.0), img_shape)
    #points /= 7.0 # Keep the values lower than 1, constant so that all the views are scaled the same way

    fig = plt.figure(figsize=(25, 10))
    plt.subplot(241)
    plt.imshow(lin_diffuse.cpu().numpy())
    plt.title(f"Linear Diffuse - it {it}")
    plt.subplot(242)
    plt.imshow(lin_specular.cpu().numpy())
    plt.title(f"Linear Specular - it {it}")
    plt.subplot(243)
    plt.imshow(linear.cpu().numpy())
    plt.title("Linear Reflectance")
    plt.subplot(245)
    plt.imshow(enh_diffuse.cpu().numpy())
    plt.title("Enhanced Diffuse")
    plt.subplot(246)
    plt.imshow(enh_specular.cpu().numpy())
    plt.title("Enhanced Specular")
    plt.subplot(247)
    plt.imshow(reflectance.cpu().numpy())
    plt.title("Enhanced Reflectance")
    plt.subplot(248)
    plt.imshow(target.cpu().numpy())
    plt.title("Target")

    if save and out_path is not None:
        plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)
        plt.close(fig)
        im = Image.fromarray((lin_diffuse.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_diffuse_it{it}.png")
        im = Image.fromarray((lin_specular.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_specular_it{it}.png")
        im = Image.fromarray((linear.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_linear_it{it}.png")
        im = Image.fromarray((reflectance.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_enh_reflectance_it{it}.png")
        im = Image.fromarray((enh_specular.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_enh_specular_it{it}.png")
        im = Image.fromarray((enh_diffuse.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_enh_diffuse_it{it}.png")
    else:
        plt.show()
            
    if wandb_act:
        wandb.log({f"{name}_view": fig}, step=it)

def validation_view_reflectance(reflectance, specular, diffuse, target, linear, img_shape, it=0, out_path=None, name="val_reflectance", save=True, wandb_act=True):
    reflectance = torch.reshape(torch.clamp(reflectance, min=0.0, max=1.0), img_shape)
    diffuse = torch.reshape(torch.clamp(diffuse, min=0.0, max=1.0), img_shape)
    target = torch.reshape(target, img_shape)
    linear = torch.reshape(torch.clamp(linear, min=0.0, max=1.0), img_shape)
    specular = torch.reshape(torch.clamp(specular, min=0.0, max=1.0), img_shape)
    
    fig = plt.figure(figsize=(25, 10))
    plt.subplot(231)
    plt.imshow(diffuse.cpu().numpy())
    plt.title(f"Diffuse - it {it}")
    plt.subplot(232)
    plt.imshow(specular.cpu().numpy())
    plt.title(f"Specular - it {it}")
    plt.subplot(233)
    plt.imshow(linear.cpu().numpy())
    plt.title("Linear mapping reflectance")
    plt.subplot(234)
    plt.imshow(reflectance.cpu().numpy())
    plt.title("Predicted reflectance")
    plt.subplot(235)
    plt.imshow(target.cpu().numpy())
    plt.title("Target reflectance")

    if save and out_path is not None:
        plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)
        plt.close(fig)
        im = Image.fromarray((diffuse.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_diffuse_it{it}.png")
        im = Image.fromarray((specular.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_specular_it{it}.png")
        im = Image.fromarray((reflectance.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_reflectance_it{it}.png")
        im = Image.fromarray((linear.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_linear_it{it}.png")
    else:
        plt.show()
            
    if wandb_act:
        wandb.log({f"{name}_view": fig}, step=it)

def validation_view_selfattention(diffuse_att, specular_att, pred_att, diffuse_kmeans, specular_kmeans, pred_kmeans, target, 
                                    img_shape, it=0, out_path=None, name="att_reflectance", save=True, wandb_act=True):
    
    diffuse_att = torch.reshape(torch.clamp(diffuse_att, min=0.0, max=1.0), img_shape)
    specular_att = torch.reshape(torch.clamp(specular_att, min=0.0, max=1.0), img_shape)
    pred_att = torch.reshape(torch.clamp(pred_att, min=0.0, max=1.0), img_shape)
    diffuse_kmeans = torch.reshape(torch.clamp(diffuse_kmeans, min=0.0, max=1.0), img_shape)
    specular_kmeans = torch.reshape(torch.clamp(specular_kmeans, min=0.0, max=1.0), img_shape)
    pred_kmeans = torch.reshape(torch.clamp(pred_kmeans, min=0.0, max=1.0), img_shape)
    target = torch.reshape(target, img_shape)
    
    fig = plt.figure(figsize=(25, 10))
    plt.subplot(241)
    plt.imshow(diffuse_att.cpu().numpy())
    plt.title(f"Self-attention Diffuse")
    plt.subplot(242)
    plt.imshow(specular_att.cpu().numpy())
    plt.title(f"Self-attention Specular")
    plt.subplot(243)
    plt.imshow(pred_att.cpu().numpy())
    plt.title("Self-attention Reflectance")
    plt.subplot(244)
    plt.imshow(target.cpu().numpy())
    plt.title("Target reflectance")
    plt.subplot(245)
    plt.imshow(diffuse_kmeans.cpu().numpy())
    plt.title(f"K-means Diffuse")
    plt.subplot(246)
    plt.imshow(specular_kmeans.cpu().numpy())
    plt.title(f"K-means Specular")
    plt.subplot(247)
    plt.imshow(pred_kmeans.cpu().numpy())
    plt.title("K-means Reflectance")

    if save and out_path is not None:
        plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)
        plt.close(fig)
        im = Image.fromarray((diffuse_att.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_diffuse_it{it}.png")
        im = Image.fromarray((specular_att.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_specular_it{it}.png")
        im = Image.fromarray((pred_att.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_it{it}.png")
        im = Image.fromarray((pred_kmeans.cpu().numpy() * 255).astype(np.uint8))
        im.save(f"{out_path}/{name}_kmeans_it{it}.png")
    else:
        plt.show()
            
    if wandb_act:
        wandb.log({f"{name}_view": fig}, step=it)


'''def validation_view_reflectance(reflectance, target, linear, img_shape, it=0, out_path=None, name="val_reflectance", save=True, wandb_act=True):
    reflectance = torch.reshape(reflectance, img_shape)
    target = torch.reshape(target, img_shape)
    linear = torch.reshape(linear, img_shape)

    fig = plt.figure(figsize=(25, 10))
    plt.subplot(131)
    plt.imshow(linear.cpu().numpy())
    plt.title("Linear mapping reflectance")
    plt.subplot(132)
    plt.imshow(reflectance.cpu().numpy())
    plt.title("Predicted reflectance")
    plt.subplot(133)
    plt.imshow(target.cpu().numpy())
    plt.title("Target reflectance")

    if save and out_path is not None:
        plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)

    if wandb_act:
        wandb.log({f"{name}_view": fig}, step=it)
        plt.close(fig)
    else:
        plt.show()
'''

def dataset_view_rgb_xnv(img, img_shape, points, normals, viewdirs):
    img = torch.reshape(img, img_shape)
    points = torch.reshape(points, img_shape)
    normals = torch.reshape(normals, img_shape)
    viewdirs = torch.reshape(viewdirs, img_shape)
    
    fig = plt.figure(figsize=(25, 4))
    plt.subplot(141)
    plt.imshow(img.numpy())
    plt.title(f"Image")
    plt.subplot(142)
    plt.imshow(points.numpy())
    plt.title("3D points")
    plt.subplot(143)
    plt.imshow(normals.numpy())
    plt.title("Normals")
    plt.subplot(144)
    plt.imshow(viewdirs.numpy())
    plt.title("View directions")
    plt.show()

def print_depths(depths, img, hwf, path="./depths.png"):
    depths = depths.reshape(hwf[0], hwf[1])
    img = img.reshape(hwf[0], hwf[1], img.shape[-1])
    
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(2,1,1)    
    ax1.imshow(img)
    ax2 = fig.add_subplot(2,2,1)   
    ax2.imshow(depths)
    plt.savefig(path)
    plt.show()

def print_normals(normals, img, hwf, path="./norms.png"):
    normals = normals.reshape(hwf[0], hwf[1], 3)
    img = img.reshape(hwf[0], hwf[1], img.shape[-1])
    
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(2,1,1)    
    ax1.imshow(img)
    ax2 = fig.add_subplot(2,2,1)   
    ax2.imshow(normals)
    plt.savefig(path)
    print(f"Saved {path}")
    plt.show()

def plot_clusters_3Dpoints(points, cluster_ids, num_clusters, colab=False, out_path=None, filename=None):
    colors = torch.zeros((points.shape[0], 3))
    
    for id in range(num_clusters):
        color = torch.rand((3,))
        colors[cluster_ids == id] = color
    
    if colab == True:
        points = points.cpu().numpy()
        colors = colors.cpu().numpy()
        ax = plt.axes(projection='3d')
        ax.view_init(-85, -15)
        ax.axis("off")
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c=colors)

        if out_path is not None:
            filename = "surface_clusters_3D.png" if filename is None else filename
            plt.savefig(f"{out_path}/{filename}", dpi=300)

        plt.show()
    else:
        import open3d as o3d

        pcd = o3d.t.geometry.PointCloud(utils.torch2open3d(points))
        pcd.point["colors"] = utils.torch2open3d(colors)
        o3d.visualization.draw_geometries([pcd.to_legacy()])


def plot_voxels_3Dpoints(points, row_ids, voxel_ids, colab=False, out_path=None, filename=None):
    colors = torch.zeros((points.shape[0], 3))
    
    for id in set(voxel_ids):
        color = torch.rand((3,))
        colors[row_ids[voxel_ids == id]] = color
    
    if colab == True:
        points = points.cpu().numpy()
        colors = colors.cpu().numpy()
        ax = plt.axes(projection='3d')
        ax.view_init(-85, -15)
        ax.axis("off")
        ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c=colors)

        if out_path is not None:
            filename = "surface_voxels_3D.png" if filename is None else filename
            plt.savefig(f"{out_path}/{filename}", dpi=300)

        plt.show()
    else:
        import open3d as o3d

        pcd = o3d.t.geometry.PointCloud(utils.torch2open3d(points))
        pcd.point["colors"] = utils.torch2open3d(colors)
        o3d.visualization.draw_geometries([pcd.to_legacy()])


def closepoints_pointcloud(points_pairs, points, colab=False, out_path="./out", filename=None):
    points_pcl = []
    colors_pcl = []

    for i in range(len(points_pairs)):
        color = np.random.uniform(size=(3,))
        points_pcl.append(points[points_pairs[i,0]].cpu().numpy())
        colors_pcl.append(color)
        points_pcl.append(points[points_pairs[i,1]].cpu().numpy())
        colors_pcl.append(color)

    points_pcl = np.stack(points_pcl)
    colors_pcl = np.stack(colors_pcl)

    if colab == True:
        ax = plt.axes(projection='3d')
        ax.view_init(-130, -130)
        ax.axis("off")
        ax.scatter(points_pcl[:,0], points_pcl[:,1], points_pcl[:,2], s=1, c=colors_pcl)

        if out_path is not None:
            fname = "surface_clusters_3D_90_-90.png" if filename is None else filename
            plt.savefig(f"{out_path}/{fname}", dpi=300)

        plt.show()
        ax = plt.axes(projection='3d')
        ax.view_init(90, 0)
        ax.axis("off")
        ax.scatter(points_pcl[:,0], points_pcl[:,1], points_pcl[:,2], s=1, c=colors_pcl)

        if out_path is not None:
            fname = "surface_clusters_3D_90_0.png" if filename is None else filename
            plt.savefig(f"{out_path}/{fname}", dpi=300)

        plt.show()
        ax = plt.axes(projection='3d')
        ax.view_init(90, 180)
        ax.axis("off")
        ax.scatter(points_pcl[:,0], points_pcl[:,1], points_pcl[:,2], s=1, c=colors_pcl)

        if out_path is not None:
            fname = "surface_clusters_3D_90_180.png" if filename is None else filename
            plt.savefig(f"{out_path}/{fname}", dpi=300)

        plt.show()
        ax = plt.axes(projection='3d')
        ax.view_init(0, 90)
        ax.axis("off")
        ax.scatter(points_pcl[:,0], points_pcl[:,1], points_pcl[:,2], s=1, c=colors_pcl)

        if out_path is not None:
            fname = "surface_clusters_3D_0_90.png" if filename is None else filename
            plt.savefig(f"{out_path}/{fname}", dpi=300)

        plt.show()
    else:
        import open3d as o3d
        import open3d.core as o3c
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points_pcl))
        pcd.point["colors"] = o3c.Tensor(colors_pcl)
        o3d.visualization.draw_geometries([pcd.to_legacy()])