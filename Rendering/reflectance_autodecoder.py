import torch
import torch.nn as nn
import numpy as np
import configargparse
import visualization as v
import gc
import copy
import wandb
from tqdm import tqdm
import sys
sys.path.append('../')


def parse_args():
    parser = configargparse.ArgumentParser(description="Initializes the geometry with a given mesh")
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    parser.add_argument('--mesh_path', type=str, help='initial mesh path')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--out_path', type=str, help='path to the output folder', default="./out")
    
    parser.add_argument('--dataset_type', type=str, help='type of dataset', choices=['synthetic', 'llff', 'tiny', 'meshroom', 'colmap'])
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--encoding_freqs', type=int, help='number of frequencies used in the positional encoding', default=6)
    parser.add_argument('--dataset_to_gpu', default=False, action="store_true")
    parser.add_argument('--load_light', action='store_true', help='load light sources positions')
    parser.add_argument("--test", action='store_true', help='use reduced number of images')

    parser.add_argument('--batch_size', type=int, default=200_000, help='number of points whose rays would be used at once')
    parser.add_argument('--num_iters', type=int, help='number of iterations to train the network', default=1000)
    parser.add_argument('--num_epochs', type=int, help='number of epochs used to learn the latent features', default=100)
    parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument('--shuffle', type=bool, default=False)
    
    parser.add_argument('--colab_path', type=str, help='google colab base dir')
    parser.add_argument('--colab', action="store_true")

    parser.add_argument("--run_id", type=str, help='Id of the run that must be resumed')
    parser.add_argument('--checkpoint_path', type=str, help='Path where checkpoints are saved')
    parser.add_argument("--resume", action='store_true', help='Resume the run from the last checkpoint')
    
    parser.add_argument('--val_images', type=int, help='number of validation images', default=100)
    parser.add_argument('--train_images', type=int, help='number of training images', default=100)
    parser.add_argument('--test_images', type=int, help='number of test images', default=100)
    
    parser.add_argument('--num_clusters', type=int, help='number of clusters of the surface points', default=10)
    parser.add_argument('--kmeans_tol', type=float, help='threshold to stop iterating the kmeans algorithm', default=1e-04)
    parser.add_argument('--kmeans_batch_size', type=int, default=200000, help='number of points to cluster at once')
    
    parser.add_argument('--latent_size', type=int, help='size of the additional features to add in the linear mapping', default=18)
    parser.add_argument('--latent_bound', type=float, help='boundaries for the latent variables', default=1.0)
    parser.add_argument('--latent_std', type=float, help='standard deviation for the initialization of the latent variables', default=1.0)
    
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
    dataset = data.NeRFDataset(args)
    dataset.compute_depths(torch.device("cpu"))
    dataset.compute_normals()
    dataset.compute_halfangles()
    
    loss_fn = nn.MSELoss()
    mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))
    
    X_NdotL_NdotH_shape = dataset.get_X_NdotL_NdotH_rgb_shape()
    embed_fn, input_ch = emb.get_embedder(in_dim=X_NdotL_NdotH_shape[-1], num_freqs=args.encoding_freqs)
    input = torch.zeros((X_NdotL_NdotH_shape[0], input_ch+args.latent_size))

    X_NdotL_NdotH, y_rgb = dataset.get_X_NdotL_NdotH_rgb("train", img=-1, device=torch.device("cpu"))
    input[:, :input_ch] = embed_fn(X_NdotL_NdotH)
    latent_features = nn.Parameter(torch.ones(X_NdotL_NdotH_shape[0], args.latent_size), requires_grad=True)
        
    decoder = net.LinearAutoDecoder(input_ch, args.latent_size, args.num_clusters)
    optimizer = torch.optim.Adam([
        {
            "params": latent_features, 
            "lr": args.lrate
        },
        {
            "params": decoder.learnable_parameters(),
            "lr": args.lrate
        }
    ])

    if args.resume:
        run = wandb.init(project="controllable-neural-rendering", 
                        entity="guillemgarrofe",
                        id=args.run_id,
                        resume=True)
        
    else:
        run = wandb.init(project="controllable-neural-rendering", 
                entity="guillemgarrofe",
                config = {
                        "learning_rate": args.lrate,
                        "num_epochs": args.num_epochs,
                        "batch_size": args.batch_size,
                        "kmeans_batch_size": args.kmeans_batch_size,
                        "dataset_type": args.dataset_type,
                        "dataset_path": args.dataset_path,
                        "num_clusters": args.num_clusters
                    })

    # INITIALIZING THE LINEAR LAYER
    epoch=0
    if wandb.run.resumed:
        wandb.restore(f"{args.checkpoint_path}/{args.run_id}.tar")
        checkpoint = torch.load(f"{args.checkpoint_path}/{args.run_id}.tar")
        decoder.load_state_dict(checkpoint['deocder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        centroids = checkpoint['centroids']
        cluster_ids = checkpoint['cluster_ids']
        epoch = checkpoint['epoch']
        print(f"Resuming {run.id} at epoch {epoch}")

    else:
        cluster_ids = torch.zeros((X_NdotL_NdotH_shape[0],), dtype=torch.long).to(device)
        centroids = torch.zeros((args.num_clusters, 3)).to(device)

        mask = (X_NdotL_NdotH[:,0] == -1.) & (X_NdotL_NdotH[:,1] == -1.) & (X_NdotL_NdotH[:,2] == -1.) #not masking takes too much time
        cluster_ids[~mask], centroids[:args.num_clusters-1] = kmeans(X_NdotL_NdotH[~mask, :3],
                                                                    num_clusters=args.num_clusters-1, 
                                                                    tol=args.kmeans_tol,
                                                                    device=device,
                                                                    batch_size=args.kmeans_batch_size)
        
        cluster_ids.masked_fill_(mask.to(device), args.num_clusters-1)
        centroids[args.num_clusters-1] = torch.tensor([-1., -1., -1.]).to(X_NdotL_NdotH)
        cluster_ids = cluster_ids.cpu()

        X_NdotL_NdotH.require_grad = False
        print("init normal")

    
    # TRAINING
    #batch_training_size = min(400_000, X_NdotL_NdotH.shape[0])
    #indices = torch.tensor(np.random.choice(np.arange(latent_features.weight.data.shape[0]), size=(batch_training_size,), replace=False))
   
    print("indices")
    pbar = tqdm(total=args.num_epochs, unit="epoch")
    pbar.update(epoch)
    while epoch < args.num_epochs:
        decoder.train()
        print("train")
        input[:, -args.latent_size:] = latent_features
        print("cat")
        linear_mappings = net.LinearAutoDecoder.compute_linear_mappings(input, 
                                                                        y_rgb, 
                                                                        cluster_ids, 
                                                                        args.num_clusters, 
                                                                        device)
        print("linear  appings")
        pred_rgb = decoder(input.to(device), cluster_ids, linear_mappings)
        print("pred rgb")
        loss = loss_fn(y_rgb.to(device), pred_rgb)
        print("loss")
        optimizer.zero_grad()
        print("zero grad")
        loss.backward()
        print("backwards")
        optimizer.step()
        print("step done")
        wandb.log({
            "tr_loss": loss,
            "tr_psnr": mse2psnr(loss)
            }, step=epoch)

        #  ------------------------ EVALUATION ------------------------
        if epoch%100 == 0:
            decoder.eval()
            i = 0
            h, w = dataset.hwf[0], dataset.hwf[1]
            print("before gettuing batch")
            X_NdotL_NdotH_i, img = dataset.get_X_NdotL_NdotH_rgb("train", img=i, device=device)
            print("have batch")
            cluster_ids_i = kmeans_predict(X_NdotL_NdotH_i[..., :3], centroids, device=device)
            print("kmeans predict done")
            input_i = torch.cat([embed_fn(X_NdotL_NdotH_i), 
                                latent_features[i*h*w:(i+1)*h*w].to(device)], dim=-1)
            print("cat done")
            pred_rgb = decoder(input_i, cluster_ids_i)
            print("decoder predict done")
            
            v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                        specular=pred_rgb.detach().cpu(), 
                                        diffuse=pred_rgb.detach().cpu(), 
                                        target=img.detach().cpu(),
                                        points=X_NdotL_NdotH_i[..., :3].detach().cpu(),
                                        linear=decoder.position_mapping(embed_fn(X_NdotL_NdotH_i), cluster_ids_i).detach().cpu(),
                                        img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                        out_path=args.out_path,
                                        it=epoch,
                                        name=f"training_reflectance_img{i}")
            
            i = np.random.randint(0, dataset.get_n_images("train"))
            X_NdotL_NdotH_i, img = dataset.get_X_NdotL_NdotH_rgb("train", img=i, device=device)
            cluster_ids_i = kmeans_predict(X_NdotL_NdotH_i[..., :3], centroids, device=device)
            input_i = torch.cat([embed_fn(X_NdotL_NdotH_i), 
                                latent_features[i*h*w:(i+1)*h*w].to(device)], dim=-1)
            pred_rgb = decoder(input_i, cluster_ids_i)

            v.validation_view_reflectance(reflectance=pred_rgb.detach().cpu(),
                                        specular=pred_rgb.detach().cpu(), 
                                        diffuse=pred_rgb.detach().cpu(), 
                                        target=img.detach().cpu(),
                                        points=X_NdotL_NdotH_i[..., :3].detach().cpu(),
                                        linear=decoder.position_mapping(embed_fn(X_NdotL_NdotH_i), cluster_ids_i).detach().cpu(),
                                        img_shape=(dataset.hwf[0], dataset.hwf[1], 3), 
                                        out_path=args.out_path,
                                        it=epoch,
                                        name=f"training_random_reflectance")

            torch.save({ # Save our checkpoint loc
                'epoch': epoch,
                'deocder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'centroids': centroids,
                'cluster_ids': cluster_ids
                }, f"{args.checkpoint_path}/{run.id}.tar")

            wandb.save(f"{args.checkpoint_path}/{run.id}.tar") # saves checkpoint to wandb    

        epoch += 1
        pbar.update(1)
    
    pbar.close()
    