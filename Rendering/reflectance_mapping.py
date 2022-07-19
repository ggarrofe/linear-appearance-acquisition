import torch
import torch.nn as nn

import configargparse
import open3d as o3d
import visualization as v

import matplotlib.pyplot as plt

import sys
sys.path.append('../')

def parse_args():
    parser = configargparse.ArgumentParser(description="Initializes the geometry with a given mesh")
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--mesh', type=str, help='initial mesh path')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--out_path', type=str, help='path to the output folder', default="./out")
    parser.add_argument('--dataset_type', type=str, help='type of dataset', choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=200_000, help='number of points whose rays would be used at once')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--N_iters', type=int, help='number of iterations to train the network', default=1000)
    parser.add_argument('--num_clusters', type=int, help='number of clusters of the surface points', default=10)
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    parser.add_argument('--colab', action="store_true")
    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument("--test", action='store_true', help='use reduced number of images')
    parser.add_argument("--resume", action='store_true', help='Resume the run from the last checkpoint')
    parser.add_argument("--run_id", type=str, help='Id of the run that must be resumed')
    parser.add_argument('--checkpoint_path', type=str, help='Path where checkpoints are saved')
    parser.add_argument('--val_images', type=int, help='number of validation images', default=100)
    parser.add_argument('--train_images', type=int, help='number of training images', default=100)
    parser.add_argument('--test_images', type=int, help='number of test images', default=100)
    parser.add_argument('--kmeans_tol', type=float, help='threshold to stop iterating the kmeans algorithm', default=1e-04)
    args = parser.parse_args()
    return args

def compute_inv(xnv, target, cluster_id, cluster_ids, embed_fn):
    mask = cluster_ids == cluster_id
    xnv_enc = embed_fn(xnv[mask])
    xnv_enc_inv = torch.linalg.pinv(xnv_enc)
    linear_mapping = xnv_enc_inv @ target[mask]
    return linear_mapping

def predict(xnv, pred, linear_mapping, cluster_id, cluster_ids, embed_fn):
    mask = (cluster_ids == cluster_id)
    if not torch.any(mask): return

    xnv_enc = embed_fn(xnv[mask])
    pred[mask] = xnv_enc @ linear_mapping
    

if __name__ == "__main__":
    args = parse_args()
    
    if args.colab:
        sys.path.append(args.colab_path)
    import utils.data as data
    import utils.networks as net
    import utils.utils as utils
    import utils.embedder as emb
    from utils.kmeans import kmeans, kmeans_predict
    
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using {device}')
     
    # Load data
    print("dataset to: ", device if args.dataset_to_gpu == True else torch.device("cpu"))
    dataset = data.NeRFDataset(args, device=device if args.dataset_to_gpu else torch.device("cpu"))

    mesh = o3d.io.read_triangle_mesh(args.mesh, print_progress=True)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    dataset.create_xnv_dataset(scene, device=device if args.dataset_to_gpu else torch.device("cpu"))
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

    for i in range(dataset.get_n_images()):
        xnv, img, depths = dataset.get_X_target("train", i, device=device)
        v.validation_view_rgb_xndv(img.detach().cpu(), 
                                    img.detach().cpu(), 
                                    points=xnv[..., :3].detach().cpu(),
                                    normals=xnv[..., 3:6].detach().cpu(),
                                    depths=depths,
                                    viewdirs=xnv[..., 6:].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    it=iter,
                                    wandb_act=False)
    sys.exit()
    # TRAINING
    embed_fn, input_ch = emb.get_embedder(in_dim=9, num_freqs=6)
    xnv, target_rgb, depths = dataset.get_tensors("train", device=device)
    
    linear_mappings = torch.zeros([args.num_clusters, input_ch, 3]).to(xnv)
    cluster_ids = torch.zeros((xnv.shape[0],)).to(xnv)
    centroids = torch.zeros((args.num_clusters, 3)).to(xnv)

    mask = (xnv[:,0] == -1.) & (xnv[:,1] == -1.) & (xnv[:,2] == -1.) #not masking takes too much time
    
    cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(X=xnv[~mask, :3],
                                                                 num_clusters=args.num_clusters-1, 
                                                                 tol=args.kmeans_tol,
                                                                 device=device,
                                                                 batch_size=args.batch_size)
    cluster_ids.masked_fill_(mask, args.num_clusters-1)
    centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(xnv)

    for cluster_id in range(args.num_clusters):
        linear_mappings[cluster_id] = compute_inv(xnv, target_rgb, cluster_id, cluster_ids, embed_fn=embed_fn)
    
    # EVALUATION
    xnv_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)
    pred_rgb_tr = torch.zeros_like(img_tr)
    colors_tr = torch.zeros_like(xnv_tr[..., :3])

    xnv_val, img_val, depths_val = dataset.get_X_target("val", 0, device=device)
    pred_rgb_val = torch.zeros_like(img_val)

    cluster_ids_tr = kmeans_predict(xnv_tr[..., :3], centroids, device=device)
    v.plot_clusters_3Dpoints(xnv_tr[..., :3], cluster_ids_tr, args.num_clusters, colab=args.colab, out_path=args.out_path, filename="train_clusters.png")

    cluster_ids_val = kmeans_predict(xnv_val[..., :3], centroids, device=device)
    v.plot_clusters_3Dpoints(xnv_val[..., :3], cluster_ids_val, args.num_clusters, colab=args.colab, out_path=args.out_path, filename="val_clusters.png")

    for cluster_id in range(args.num_clusters):
        predict(xnv_tr, pred_rgb_tr, linear_mappings[cluster_id], cluster_id, cluster_ids_tr, embed_fn)
        predict(xnv_val, pred_rgb_val, linear_mappings[cluster_id], cluster_id, cluster_ids_val, embed_fn)
    
    loss_tr = loss_fn(pred_rgb_tr, img_tr)
    loss_val = loss_fn(pred_rgb_val, img_val)

    print({
            "loss_tr": loss_tr,
            "psnr_tr": mse2psnr(loss_tr),
            "loss_val": loss_val,
            "psnr_val": mse2psnr(loss_val)
            })

    v.validation_view_rgb_xndv(pred_rgb_tr.detach().cpu(), 
                                    img_tr.detach().cpu(), 
                                    points=xnv_tr[..., :3].detach().cpu(),
                                    normals=xnv_tr[..., 3:6].detach().cpu(),
                                    depths=depths_tr,
                                    viewdirs=xnv_tr[..., 6:].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    it=args.num_clusters, 
                                    out_path=args.out_path,
                                    name="training_xnv",
                                    wandb_act=False)

    v.validation_view_rgb_xndv(pred_rgb_val.detach().cpu(), 
                                    img_val.detach().cpu(), 
                                    points=xnv_val[..., :3].detach().cpu(),
                                    normals=xnv_val[..., 3:6].detach().cpu(),
                                    depths=depths_val,
                                    viewdirs=xnv_val[..., 6:].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    it=args.num_clusters, 
                                    out_path=args.out_path,
                                    name="val_xnv",
                                    wandb_act=False)