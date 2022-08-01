import torch
import torch.nn as nn

import configargparse
import open3d as o3d
import visualization as v
from tqdm import tqdm
import gc

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

def compute_inv(xh, target, cluster_id, cluster_ids, embed_fn, device=torch.device("cuda"), batch_size=1e07):
    mask = cluster_ids == cluster_id
    xh, target = xh[mask], target[mask]
    del mask
    gc.collect()

    if xh.shape[0] < batch_size:
        xh_enc_inv = torch.linalg.pinv(embed_fn(xh.to(device)))

        linear_mapping = xh_enc_inv @ target.to(device)
    else: 
        xh, indices = utils.filter_duplicates(xh)
        target = target[indices]
        xh_enc_inv = torch.linalg.pinv(embed_fn(xh))
        linear_mapping = xh_enc_inv @ target
    return linear_mapping.T.to(device)

def predict(xh, pred, spec_pred, diffuse_pred, linear_mapping, cluster_id, cluster_ids, embed_fn):
    if not torch.any(cluster_ids == cluster_id): return

    xh_enc = embed_fn(xh[cluster_ids == cluster_id])
    pred[cluster_ids == cluster_id] = xh_enc @ linear_mapping.T

    # First half of the linear mapping will predict the diffuse color (only depends on the position)
    diffuse_pred[cluster_ids == cluster_id] = xh_enc[..., :48] @ linear_mapping[..., :48].T

    # Second half of the linear mapping will predict the specular color (depends on the half-angle vector)
    linear_mapping_spec = torch.cat([linear_mapping[..., :36], linear_mapping[..., 48:]], dim=-1)
    spec_pred[cluster_ids == cluster_id] = torch.cat([xh_enc[..., :36], xh_enc[..., 48:]], dim=-1) @ linear_mapping_spec.T



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
    dataset.compute_depths(scene, device=torch.device("cpu"))
    dataset.compute_normals()
    dataset.compute_halfangles()
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    
    # TRAINING
    x_NdotL_NdotH, target_rgb = dataset.get_X_NdotL_NdotH_rgb("train", img=-1, device=torch.device("cpu"))
    embed_fn, input_ch = emb.get_embedder(in_dim=x_NdotL_NdotH.shape[-1], num_freqs=6)
    
    linear_mappings = torch.zeros([args.num_clusters, 3, input_ch]).to(device)
    cluster_ids = torch.zeros((x_NdotL_NdotH.shape[0],),).to(device)
    centroids = torch.zeros((args.num_clusters, 3)).to(device)

    mask = (x_NdotL_NdotH[:,0] == -1.) & (x_NdotL_NdotH[:,1] == -1.) & (x_NdotL_NdotH[:,2] == -1.) #not masking takes too much time
    cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(x_NdotL_NdotH[~mask, :3],
                                                                 num_clusters=args.num_clusters-1, 
                                                                 tol=args.kmeans_tol,
                                                                 device=device,
                                                                 batch_size=args.batch_size)
    
    cluster_ids.masked_fill_(mask.to(device), args.num_clusters-1)
    centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(x_NdotL_NdotH)
    cluster_ids = cluster_ids.cpu()

    for cluster_id in tqdm(range(args.num_clusters), unit="linear mapping", desc="Computing linear mappings"):
        linear_mappings[cluster_id] = compute_inv(x_NdotL_NdotH, 
                                                  target_rgb, 
                                                  cluster_id, 
                                                  cluster_ids, 
                                                  embed_fn=embed_fn,
                                                  device=device)

    linear_net = net.LinearNetwork(in_features=x_NdotL_NdotH.shape[-1], linear_mappings=linear_mappings, num_freqs=6)
    linear_net.to(device)
    
    # EVALUATION
    print("evaluating...")
    for i in range(5):
        x_NdotL_NdotH, img_tr = dataset.get_X_NdotL_NdotH_rgb("train", img=i, device=device)
        
        cluster_ids_tr = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
        pred_rgb = linear_net(x_NdotL_NdotH, cluster_ids_tr)
        pred_rgb_spec = linear_net.specular(x_NdotL_NdotH, cluster_ids_tr)
        pred_rgb_diff = linear_net.diffuse(x_NdotL_NdotH, cluster_ids_tr)
        
        spec_comp = linear_net.specular_component(x_NdotL_NdotH, cluster_ids_tr)
        diff_comp = linear_net.diffuse_component(x_NdotL_NdotH, cluster_ids_tr)
        amb_comp = linear_net.ambient_component(x_NdotL_NdotH, cluster_ids_tr)
          
        loss_tr = loss_fn(pred_rgb, img_tr)

        v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                    specular=pred_rgb_spec.detach().cpu(), 
                                    diffuse=pred_rgb_diff.detach().cpu(), 
                                    target=img_tr.detach().cpu(),
                                    points=x_NdotL_NdotH[..., :3].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    out_path=args.out_path,
                                    it=i,
                                    name="training_reflectance",
                                    wandb_act=False)
        
        v.validation_view_reflectance(reflectance=(amb_comp+spec_comp+diff_comp).detach().cpu(),
                                    specular=(amb_comp+spec_comp).detach().cpu(), 
                                    diffuse=(amb_comp+diff_comp).detach().cpu(), 
                                    target=img_tr.detach().cpu(),
                                    points=x_NdotL_NdotH[..., :3].detach().cpu(),
                                    img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                    out_path=args.out_path,
                                    it=i,
                                    name="training_reflectance_test",
                                    wandb_act=False)

    for i in range(dataset.get_n_images("val")):
        x_NdotL_NdotH, img_val = dataset.get_X_NdotL_NdotH_rgb("val", img=i, device=device)
        cluster_ids_tr = kmeans_predict(x_NdotL_NdotH[..., :3], centroids, device=device)
        pred_rgb = linear_net(x_NdotL_NdotH, cluster_ids_tr)
        pred_rgb_spec = linear_net.specular(x_NdotL_NdotH, cluster_ids_tr)
        pred_rgb_diff = linear_net.diffuse(x_NdotL_NdotH, cluster_ids_tr)

        v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                      specular=pred_rgb_spec.detach().cpu(),
                                      diffuse=pred_rgb_diff.detach().cpu(),
                                      target=img_val.detach().cpu(),
                                      points=x_NdotL_NdotH[..., :3].detach().cpu(),
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