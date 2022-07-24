import torch
import torch.nn as nn

import configargparse
import open3d as o3d
import visualization as v

import matplotlib.pyplot as plt

import sys
sys.path.append('../')

print(sys.path)

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

def filter_duplicates(xh, target):
    unique_xh, inverse = torch.unique(xh, sorted=True, return_inverse=True, dim=0)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    inidices = inverse.new_empty(unique_xh.size(0)).scatter_(0, inverse, perm)

    return unique_xh, target[inidices]

def compute_inv(xh, target, cluster_id, cluster_ids, embed_fn, device=torch.device("cuda"), batch_size=1e07):
    mask = cluster_ids == cluster_id
    xh, target, cluster_ids = xh[mask], target[mask], cluster_ids[mask]
    
    # if the target is the same and the input is the same, reduce it
    xh, target = filter_duplicates(xh, target)

    if xh.shape[0] < batch_size:
        xh_enc_inv = torch.linalg.pinv(embed_fn(xh.to(device)))
        linear_mapping = xh_enc_inv @ target.to(device)
    else: 
        print(f"Inverse computed in gpu {xh.shape[0]}")
        xh_enc_inv = torch.linalg.pinv(embed_fn(xh))
        linear_mapping = xh_enc_inv @ target
    return linear_mapping.to(device)

def predict(xh, pred, spec_pred, diffuse_pred, linear_mapping, cluster_id, cluster_ids, embed_fn):
    if not torch.any(cluster_ids == cluster_id): return

    xh_enc = embed_fn(xh[cluster_ids == cluster_id])
    pred[cluster_ids == cluster_id] = xh_enc @ linear_mapping

    # First half of the linear mapping will predict the diffuse color (only depends on the position)
    diffuse_pred[cluster_ids == cluster_id] = xh_enc[..., :int(xh_enc.shape[1]/2)] @ linear_mapping[:int(linear_mapping.shape[0]/2)]

    # Second half of the linear mapping will predict the specular color (depends on the half-angle vector)
    spec_pred[cluster_ids == cluster_id] = xh_enc[..., int(xh_enc.shape[1]/2):] @ linear_mapping[int(linear_mapping.shape[0]/2):]

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
    dataset.compute_xh()
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    
    # TRAINING
    embed_fn, input_ch = emb.get_embedder(in_dim=6, num_freqs=6)
    xh, target_rgb = dataset.get_xh_rgb("train", img=-1, device=torch.device("cpu"))
    
    linear_mappings = torch.zeros([args.num_clusters, input_ch, 3]).to(device)
    cluster_ids = torch.zeros((xh.shape[0],),).to(device)
    centroids = torch.zeros((args.num_clusters, 3)).to(device)

    mask = (xh[:,0] == -1.) & (xh[:,1] == -1.) & (xh[:,2] == -1.) #not masking takes too much time
    cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(xh[~mask, :3],
                                                                 num_clusters=args.num_clusters-1, 
                                                                 tol=args.kmeans_tol,
                                                                 device=device,
                                                                 batch_size=args.batch_size)
    
    cluster_ids.masked_fill_(mask.to(device), args.num_clusters-1)
    centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(xh)
    cluster_ids = cluster_ids.cpu()

    for cluster_id in range(args.num_clusters):
        linear_mappings[cluster_id] = compute_inv(torch.cat([xh[..., :3], xh[..., -3:]], dim=-1), 
                                                  target_rgb, 
                                                  cluster_id, 
                                                  cluster_ids, 
                                                  embed_fn=embed_fn,
                                                  device=device)
    
    # EVALUATION
    xh, img_tr = dataset.get_xh_rgb("train", img=0, device=device)
    print("img tr", img_tr.shape)
    points_H_tr = torch.cat([xh[..., :3], xh[..., -3:]], dim=-1)
    pred_rgb = torch.zeros_like(img_tr)
    pred_rgb_spec = torch.zeros_like(img_tr)
    pred_rgb_diff = torch.zeros_like(img_tr)

    cluster_ids_tr = kmeans_predict(points_H_tr[..., :3], centroids, device=device)

    for cluster_id in range(args.num_clusters):
        predict(points_H_tr, pred_rgb, pred_rgb_spec, pred_rgb_diff, linear_mappings[cluster_id], cluster_id, cluster_ids_tr, embed_fn)
    
    v.plot_clusters_3Dpoints(points_H_tr[..., :3], cluster_ids_tr, args.num_clusters, colab=args.colab, out_path=args.out_path, filename="train_clusters.png")
        
    loss_tr = loss_fn(pred_rgb, img_tr)

    v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                  specular=pred_rgb_spec.detach().cpu(), 
                                  diffuse=pred_rgb_diff.detach().cpu(), 
                                  target=img_tr.detach().cpu(),
                                  points=xh[..., :3].detach().cpu(),
                                  img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                  out_path=args.out_path,
                                  name="training_reflectance",
                                  wandb_act=False)

    for i in range(dataset.get_n_images("val")):
        xh, img_val = dataset.get_xh_rgb("val", img=i, device=device)
        points_H_val = torch.cat([xh[..., :3], xh[..., -3:]], dim=-1)

        cluster_ids_val = kmeans_predict(points_H_val[..., :3], centroids, device=device)
        for cluster_id in range(args.num_clusters):
            predict(points_H_val, pred_rgb, pred_rgb_spec, pred_rgb_diff, linear_mappings[cluster_id], cluster_id, cluster_ids_val, embed_fn)
        
        v.plot_clusters_3Dpoints(points_H_val[..., :3], cluster_ids_val, args.num_clusters, colab=args.colab, out_path=args.out_path, filename="val_clusters.png")

        v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                      specular=pred_rgb_spec.detach().cpu(),
                                      diffuse=pred_rgb_diff.detach().cpu(),
                                      target=img_val.detach().cpu(),
                                      points=xh[..., :3].detach().cpu(),
                                      it=i,
                                      img_shape=(dataset.hwf[0], dataset.hwf[1], 3),
                                      out_path=args.out_path,
                                      name="val_reflectance",
                                      wandb_act=False)
        
        loss_val = loss_fn(pred_rgb, img_val)

    print({
            "loss_tr": loss_tr,
            "psnr_tr": mse2psnr(loss_tr),
            "loss_val": loss_val,
            "psnr_val": mse2psnr(loss_val)
            })