from multiprocessing import Semaphore

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import gc
import nvidia_smi
from torch_geometric.nn import knn
from threading import * 

import scipy.spatial

def print_memory_usage():
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"Device {0}: {nvidia_smi.nvmlDeviceGetName(handle)}, Memory : ({100*info.free/info.total:.2f}% free): {info.total} (total), {info.free} (free), {info.used} (used)")


def init(X, num_clusters, seed):
    """Initializes the centroids for the first kmeans iteration.

    Args:
        X (torch.tensor): data from where we will create the clusters
        num_clusters (int): number of clusters that we will divide the data into
        seed (int or 1-d array_like): random seed

    Returns:
        torch.tensor: cluster centroids
    """
    if seed is not None:
        np.random.seed(seed)
        
    indices = np.random.choice(X.shape[0], num_clusters, replace=False)
    centroids = X[indices]
    return centroids

def kmeans_lite(
        X,
        num_clusters,
        centroids=None,
        tol=1e-4,
        iter_limit=0,
        device=torch.device('cpu'),
        seed=None,
        batch_size=None
):
    X_device = X.device
    X = X.float().to(device)

    if centroids is None:
        centroids = init(X, num_clusters, seed=seed)

    if batch_size is None or batch_size > len(X):
        batch_size = len(X)

    cluster_ids = torch.zeros((X.shape[0],), dtype=torch.long).to(X.device)
    means = torch.zeros(num_clusters, X.shape[-1]).to(X)
    n_samples = torch.zeros(num_clusters, 1).to(X)

    iteration = 0
    pbar = tqdm(desc=f'Running kmeans on {device}', unit="it")
    while True:
        centroids_pre = centroids.clone()
        
        means[:,:] = 0.0
        n_samples[:] = 0
        
        for i in range(0, len(X), batch_size):
            # Assign each training example to the nearest centroid
            dis = distance(X[i:i+batch_size], centroids)
            cluster_ids[i:i+batch_size] = torch.argmin(dis, dim=1)

            # Update the online mean value
            for cluster in range(num_clusters):
                new_samples = X[i:i+batch_size][cluster_ids[i:i+batch_size] == cluster].shape[0]

                if new_samples > 0:
                    sum = X[i:i+batch_size][cluster_ids[i:i+batch_size] == cluster].sum(dim=0)
                    n_samples[cluster] += new_samples
                    means[cluster] += (sum - new_samples*means[cluster])/n_samples[cluster]

        # Avoids converging to only one cluster
        for cluster in range(num_clusters):
            randint = torch.randint(len(X), (1,))
            if n_samples[cluster] == 0:
                means[cluster] = X[randint]

        centroids = means.clone()
        center_shift = torch.norm(centroids-centroids_pre, dim=1).sum()
        iteration += 1

        pbar.set_postfix(
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}',
            num_clusters=f'{num_clusters}'
        )
        pbar.update(1)

        # Convergence check
        if center_shift ** 2 < tol:
            break

        if iter_limit != 0 and iteration >= iter_limit:
            break

    pbar.close()

    # Free up GPU
    del means
    del n_samples
    del center_shift
    del dis
    cluster_ids = cluster_ids.cpu()
    centroids = centroids.cpu()
    X = X.to(X_device)
    torch.cuda.empty_cache()

    return cluster_ids, centroids


def kmeans_lite_predict(
        X,
        centroids,
        device=torch.device('cpu')
):
    X = X.float().to(device)
    dis = distance(X, centroids.to(device))
    return torch.argmin(dis, dim=1)


def distance(points, centroids):
    # batch_size x 1 x N
    points = points.unsqueeze(dim=1)

    # 1 x num_clusters x N
    centroids = centroids.unsqueeze(dim=0)

    # broadcasted subtract
    dis = (points - centroids) ** 2.0

    # batch_size x num_clusters matrix 
    return dis.sum(dim=-1).squeeze()

def k_dist(x, y, k):
    tree = scipy.spatial.cKDTree(x.detach().numpy())
    dist, col = tree.query(
        y.detach().cpu(), k=k, distance_upper_bound=x.size(1))
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    mask = 1 - torch.isinf(dist).view(-1)
    row, col = row.view(-1)[mask], col.view(-1)[mask]

    return row, col, dist

def kmeans(
        X,
        num_clusters,
        centroids=None,
        tol=1e-4,
        iter_limit=0,
        device=torch.device('cpu'),
        seed=None,
        batch_size=None
):
    X_device = X.device
    X = X.float().to(device)

    if centroids is None:
        centroids = init(X, num_clusters, seed=seed)

    if batch_size is None or batch_size > len(X):
        batch_size = len(X)

    cluster_ids = torch.zeros((X.shape[0],), dtype=torch.long).to(X.device)
    means = torch.zeros(num_clusters, X.shape[-1]).to(X)
    n_samples = torch.zeros(num_clusters, 1).to(X)

    iteration = 0
    pbar = tqdm(desc=f'Running kmeans on {device}', unit="it")
    while True:
        centroids_pre = centroids.clone()
        
        means[:,:] = 0.0
        n_samples[:] = 0
        
        for i in range(0, len(X), batch_size):
            # Assign each training example to the nearest centroid
            nearest = knn(centroids, X[i:i+batch_size], k=1)
            rows = nearest[0] + i
            cluster_ids[rows] = nearest[1]

            # Update the online mean value
            for cluster in range(num_clusters):
                new_samples = X[i:i+batch_size][cluster_ids[i:i+batch_size] == cluster].shape[0]

                if new_samples > 0:
                    sum = X[i:i+batch_size][cluster_ids[i:i+batch_size] == cluster].sum(dim=0)
                    n_samples[cluster] += new_samples
                    means[cluster] += (sum - new_samples*means[cluster])/n_samples[cluster]

        # Avoids converging to only one cluster
        for cluster in range(num_clusters):
            randint = torch.randint(len(X), (1,))
            if n_samples[cluster] == 0:
                means[cluster] = X[randint]

        centroids = means.clone()
        center_shift = torch.norm(centroids-centroids_pre, dim=1).sum()
        iteration += 1

        pbar.set_postfix(
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}',
            num_clusters=f'{num_clusters}'
        )
        pbar.update(1)

        # Convergence check
        if center_shift ** 2 < tol:
            break

        if iter_limit != 0 and iteration >= iter_limit:
            break

    pbar.close()

    # Free up GPU
    del means
    del n_samples
    del center_shift
    cluster_ids = cluster_ids.cpu()
    centroids = centroids.cpu()
    X = X.to(X_device)
    torch.cuda.empty_cache()

    return cluster_ids, centroids

def kmeans_predict(
        X,
        centroids,
        device=torch.device('cpu'),
        k=3
):
    X_device = X.device
    '''cluster_ids = torch.zeros((len(X)), device=X.device, dtype=torch.long)
    nearest = knn(centroids.to(device), X.to(device), 3)
    rows = nearest[0].to(cluster_ids)
    cluster_ids[rows] = nearest[1].to(cluster_ids)'''
    X = X.to(device)
    centroids = centroids.to(device)
    nearest = knn(centroids, X, k)
    cluster_ids = nearest[1].reshape(-1, k).to(X_device)

    dist = (X-centroids[cluster_ids.T]).pow(2).sum(2).sqrt()
    #suma = dist.sum(0)
    #lambda_val = 15.0
    #weights = torch.nn.functional.softmax(lambda_val*torch.exp(-lambda_val*dist), dim=0)
    return cluster_ids, dist

def self_distances(centroids):
    tree = scipy.spatial.cKDTree(centroids.detach().numpy())
    dist, col = tree.query(centroids.detach().cpu(), k=len(centroids))
    dist = torch.from_numpy(dist).to(centroids.dtype)
    col = torch.from_numpy(col).to(torch.long)
    dist2 = torch.zeros_like(dist)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, len(centroids))
    col_d = torch.arange(col.size(0), dtype=torch.long).view(1, -1).repeat(1, len(centroids))
    dist2[row.flatten(), col.flatten()] = dist[row.flatten(), col_d.flatten()]
    return dist2

def nearest_distances(centroids, y, k):
    tree = scipy.spatial.cKDTree(centroids.detach().cpu().numpy())
    dist, col = tree.query(
        y.detach().cpu(), k=k)
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)

    return col, dist

# --------------- Threaded version --------------------

class SharedInt(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SharedInt, cls).__new__(cls)
    return cls.instance

class SharedMeans(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SharedInt, cls).__new__(cls)
    return cls.instance

class SharedNumSamples(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SharedInt, cls).__new__(cls)
    return cls.instance

class SharedIds(object):
  def __new__(cls):
    if not hasattr(cls, 'instance'):
      cls.instance = super(SharedInt, cls).__new__(cls)
    return cls.instance

def update_means(X,
                 centroids,
                 num_clusters,
                 semaphore,
                 thread_id,
                 batch_size=None):
    
    shared_i = SharedInt()
    shared_ids = SharedIds()
    shared_means = SharedMeans()
    shared_num_samples = SharedNumSamples()

    while True:
        semaphore.acquire()
        if shared_i.val < len(X):
            print(f"Thread {thread_id} working on data from {i} to {i+batch_size}.")
            i = shared_i.val
            shared_i.val = shared_i.val + batch_size 
        else:
            return
        semaphore.release()

        # Assign each training example to the nearest centroid
        nearest = knn(centroids, X[i:i+batch_size], k=1)
        rows = nearest[0] + i
        shared_ids.val[rows] = nearest[1]

        # Update the online mean value
        for cluster in range(num_clusters):
            new_samples = X[i:i+batch_size][shared_ids.val[i:i+batch_size] == cluster].shape[0]

            if new_samples > 0:
                sum = X[i:i+batch_size][shared_ids.val[i:i+batch_size] == cluster].sum(dim=0)
                shared_num_samples.val[cluster] += new_samples
                shared_means.val[cluster] += (sum - new_samples*shared_means.val[cluster])/shared_num_samples.val[cluster]

    

def kmeans_threaded(
        X,
        num_clusters,
        centroids=None,
        tol=1e-4,
        iter_limit=0,
        device=torch.device('cpu'),
        seed=None,
        batch_size=None,
        num_threads=1
):
    X_device = X.device
    X = X.float().to(device)

    if centroids is None:
        centroids = init(X, num_clusters, seed=seed)

    if batch_size is None or batch_size > len(X):
        batch_size = len(X)


    semaphore = Semaphore(1) 
    shared_i = SharedInt()
    shared_ids = SharedIds()
    shared_means = SharedMeans()
    shared_num_samples = SharedNumSamples()
    shared_ids.val = torch.zeros((X.shape[0],), dtype=torch.long).to(X.device)
    shared_means.val = torch.zeros(num_clusters, X.shape[-1]).to(X)
    shared_num_samples.val = torch.zeros(num_clusters, 1).to(X)

    iteration = 0
    pbar = tqdm(desc=f'Running kmeans on {device}', unit="it")

    while True:
        centroids_pre = centroids.clone()
        
        shared_means.val[:,:] = 0.0
        shared_num_samples.val[:] = 0
        shared_i.val = 0
        threads = []
        for thread_id in range(num_threads):
            threads.append(Thread(target = update_means, args = (X, centroids, num_clusters, semaphore, thread_id, batch_size)))
            [thread.start() for thread in threads]
            [thread.join() for thread in threads]
        
        # Avoids converging to only one cluster
        for cluster in range(num_clusters):
            randint = torch.randint(len(X), (1,))
            if shared_num_samples.val[cluster] == 0:
                shared_means.val[cluster] = X[randint]

        centroids = shared_means.val.clone()
        center_shift = torch.norm(centroids-centroids_pre, dim=1).sum()
        iteration += 1

        pbar.set_postfix(
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}',
            num_clusters=f'{num_clusters}'
        )
        pbar.update(1)

        # Convergence check
        if center_shift ** 2 < tol:
            break

        if iter_limit != 0 and iteration >= iter_limit:
            break

    pbar.close()

    # Free up GPU
    del shared_means.val
    del shared_num_samples.val
    del center_shift
    shared_ids.val = shared_ids.val.cpu()
    centroids = centroids.cpu()
    X = X.to(X_device)
    torch.cuda.empty_cache()

    return shared_ids.val, centroids