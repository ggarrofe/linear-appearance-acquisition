import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import wandb

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

    wandb.log({f"{name}_view": fig})
    if out_path is not None:
        plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)

def validation_view_rgb_xndv(rgb_map, val_target, img_shape, points, normals, depths, viewdirs, it=0, out_path=None, name="validation_xnv", save=True):
    rgb_map = torch.reshape(rgb_map, img_shape)
    val_target = torch.reshape(val_target, img_shape)
    points = torch.reshape(points, img_shape)
    points /= 7.0 # Keep the values lower than 1, constant so that all the views are scaled the same way
    normals = torch.reshape(normals, img_shape)
    viewdirs = torch.reshape(viewdirs, img_shape)
    depths = torch.reshape(depths, img_shape)
    depths /= torch.max(depths)

    fig = plt.figure(figsize=(25, 10))
    plt.subplot(231)
    plt.imshow(rgb_map.numpy())
    plt.title(f"Reconstruction - it {it}")
    plt.subplot(232)
    plt.imshow(val_target.numpy())
    plt.title("Target image")
    plt.subplot(233)
    plt.imshow(points.numpy())
    plt.title("3D points")
    plt.subplot(234)
    plt.imshow(normals.numpy())
    plt.title("Normals")
    plt.subplot(235)
    plt.imshow(viewdirs.numpy())
    plt.title("View directions")
    plt.subplot(236)
    plt.imshow(depths.numpy())
    plt.title("Depths")

    if save:
        wandb.log({f"{name}_view": fig})
        if out_path is not None:
            plt.savefig(f"{out_path}/{name}_it{it}.png", bbox_inches='tight', dpi=150)

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