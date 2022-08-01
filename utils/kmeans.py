from functools import partial

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


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
    return cluster_ids, centroids


def kmeans_predict(
        X,
        centroids,
        device=torch.device('cpu')
):
    X = X.float().to(device)
    dis = distance(X, centroids)
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
