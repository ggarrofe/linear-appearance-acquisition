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
    parser.add_argument('--load_light', action='store_true', help='load light sources positions')
    args = parser.parse_args()
    return args

def compute_inv(xh, target, cluster_id, cluster_ids, embed_fn):
    mask = cluster_ids == cluster_id
    xh = embed_fn(xh[mask])
    xh_enc_inv = torch.linalg.pinv(xh)
    linear_mapping = xh_enc_inv @ target[mask]
    return linear_mapping

def predict(xh, pred, spec_pred, diffuse_pred, linear_mapping, cluster_id, cluster_ids, embed_fn):
    mask = (cluster_ids == cluster_id)
    if not torch.any(mask): return

    xh_enc = embed_fn(xh[mask])
    pred[mask] = xh_enc @ linear_mapping

    # First half of the linear mapping will predict the diffuse color (only depends on the position)
    diffuse_pred[mask] = xh_enc[..., :int(xh_enc.shape[1]/2)] @ linear_mapping[:int(linear_mapping.shape[0]/2)]

    # Second half of the linear mapping will predict the specular color (depends on the half-angle vector)
    spec_pred[mask] = xh_enc[..., int(xh_enc.shape[1]/2):] @ linear_mapping[int(linear_mapping.shape[0]/2):]

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
    dataset.compute_VLH()
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

    if args.test:
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
    
    # TRAINING
    embed_fn, input_ch = emb.get_embedder(in_dim=6, num_freqs=6)
    _, target_rgb, depths = dataset.get_tensors("train", device=device) #_ is xnv
    points_VLH = dataset.get_points_VLH("train", img=-1, device=device)
    
    linear_mappings = torch.zeros([args.num_clusters, input_ch, 3]).to(points_VLH)
    cluster_ids = torch.zeros((points_VLH.shape[0],),).to(points_VLH)
    centroids = torch.zeros((args.num_clusters, 3)).to(points_VLH)

    mask = (points_VLH[:,0] == -1.) & (points_VLH[:,1] == -1.) & (points_VLH[:,2] == -1.) #not masking takes too much time
    cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(points_VLH[~mask, :3],
                                                                 num_clusters=args.num_clusters-1, 
                                                                 tol=args.kmeans_tol,
                                                                 device=device,
                                                                 batch_size=args.batch_size)
    # prob cluster ids is double and num_clusters no
    cluster_ids.masked_fill_(mask, args.num_clusters-1)
    centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(points_VLH)

    for cluster_id in range(args.num_clusters):
        linear_mappings[cluster_id] = compute_inv(torch.cat([points_VLH[..., :3], points_VLH[..., -3:]], dim=-1), 
                                                  target_rgb, 
                                                  cluster_id, 
                                                  cluster_ids, 
                                                  embed_fn=embed_fn)
    
    # EVALUATION
    xnv_tr, img_tr, depths_tr = dataset.get_X_target("train", 0, device=device)
    points_VLH = dataset.get_points_VLH("train", img=0, device=device)
    points_H_tr = torch.cat([points_VLH[..., :3], points_VLH[..., -3:]], dim=-1)
    pred_rgb_tr = torch.zeros_like(img_tr)
    pred_rgb_spec_tr = torch.zeros_like(img_tr)
    pred_rgb_diff_tr = torch.zeros_like(img_tr)

    xnv_val, img_val, depths_val = dataset.get_X_target("val", 0, device=device)
    points_VLH = dataset.get_points_VLH("val", img=0, device=device)
    points_H_val = torch.cat([points_VLH[..., :3], points_VLH[..., -3:]], dim=-1)
    pred_rgb_val = torch.zeros_like(img_val)
    pred_rgb_spec_val = torch.zeros_like(img_val)
    pred_rgb_diff_val = torch.zeros_like(img_val)

    cluster_ids_tr = kmeans_predict(points_H_tr[..., :3], centroids, device=device)
    v.plot_clusters_3Dpoints(points_H_tr[..., :3], cluster_ids_tr, args.num_clusters, colab=args.colab, out_path=args.out_path, filename="train_clusters.png")

    cluster_ids_val = kmeans_predict(points_H_val[..., :3], centroids, device=device)
    v.plot_clusters_3Dpoints(points_H_val[..., :3], cluster_ids_val, args.num_clusters, colab=args.colab, out_path=args.out_path, filename="val_clusters.png")

    for cluster_id in range(args.num_clusters):
        predict(points_H_tr, pred_rgb_tr, pred_rgb_spec_tr, pred_rgb_diff_tr, linear_mappings[cluster_id], cluster_id, cluster_ids_tr, embed_fn)
        predict(points_H_val, pred_rgb_val, pred_rgb_spec_val, pred_rgb_diff_val, linear_mappings[cluster_id], cluster_id, cluster_ids_val, embed_fn)
    
    loss_tr = loss_fn(pred_rgb_tr, img_tr)
    loss_val = loss_fn(pred_rgb_val, img_val)

    print({
            "loss_tr": loss_tr,
            "psnr_tr": mse2psnr(loss_tr),
            "loss_val": loss_val,
            "psnr_val": mse2psnr(loss_val)
            })

    v.validation_view_reflectance(reflectance=pred_rgb_tr.detach().cpu(),
                                  specular=pred_rgb_spec_tr.detach().cpu(), 
                                  diffuse=pred_rgb_diff_tr.detach().cpu(), 
                                  target=img_tr.detach().cpu(),
                                  points=points_VLH[..., :3].detach().cpu(),
                                  normals=xnv_tr[..., 3:6].detach().cpu(),
                                  depths=depths_tr,
                                  img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                  out_path=args.out_path,
                                  name="training_reflectance",
                                  wandb_act=False)

    v.validation_view_reflectance(reflectance=pred_rgb_val.detach().cpu(),
                                  specular=pred_rgb_spec_val.detach().cpu(), 
                                  diffuse=pred_rgb_diff_val.detach().cpu(), 
                                  target=img_val.detach().cpu(),
                                  points=points_VLH[..., :3].detach().cpu(),
                                  normals=xnv_val[..., 3:6].detach().cpu(),
                                  depths=depths_val,
                                  img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                  out_path=args.out_path,
                                  name="val_reflectance",
                                  wandb_act=False)