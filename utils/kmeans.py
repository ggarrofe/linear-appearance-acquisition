from functools import partial

import numpy as np
import torch
from tqdm import tqdm


def initialize(X, num_clusters, seed):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param seed: (int) seed for kmeans
    :return: (np.array) initial state
    """
    num_samples = len(X)
    if seed is not None:
        np.random.seed(seed)
        
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    centroids = X[indices]
    return centroids


def kmeans_old(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        tqdm_flag=True,
        iter_limit=0,
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001,
        seed=None,
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param tqdm_flag: Allows to turn logs on and off
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if tqdm_flag:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device, tqdm_flag=tqdm_flag)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    elif distance == 'soft_dtw':
        sdtw = SoftDTW(use_cuda=device.type == 'cuda', gamma=gamma_for_soft_dtw)
        pairwise_distance_function = partial(pairwise_soft_dtw, sdtw=sdtw, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters, seed=seed)
    else:
        if tqdm_flag:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if tqdm_flag:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:

        dis = pairwise_distance_function(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            # https://github.com/subhadarship/kmeans_pytorch/issues/16
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]

            initial_state[index] = selected.mean(dim=0)

        print("old", initial_state)
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        if tqdm_flag:
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()

        break
        if center_shift ** 2 < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return choice_cluster.cpu(), initial_state.cpu()

def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        centroids=None,
        tol=1e-4,
        iter_limit=0,
        device=torch.device('cpu'),
        seed=None,
        batch_size=200000
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param seed: (int) seed for kmeans
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :param iter_limit: hard limit for max number of iterations
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    else:
        raise NotImplementedError

    X = X.float().to(device)

    # initialize
    if centroids is None:  # ToDo: make this less annoyingly weird
        centroids = initialize(X, num_clusters, seed=seed)

    if batch_size > len(X):
        batch_size = len(X)

    iteration = 0
    pbar = tqdm(desc='Running kmeans on {device}', unit="it", leave=False)
    means = torch.zeros(num_clusters, X.shape[-1]).to(X)
    nearest_centroid = torch.zeros((X.shape[0],)).to(X)
    n_samples = torch.zeros(num_clusters, 1).to(X)

    while True:
        centroids_pre = centroids.clone()
        means[:,:] = 0.0
        
        for i in range(0, len(X), batch_size):#, leave=False, unit="batch", desc=f"Running iteration {iteration}"):
            # Assign each training example to the nearest centroid
            dis = pairwise_distance_function(X[i:i+batch_size], centroids)
            nearest_centroid[i:i+batch_size] = torch.argmin(dis, dim=1)

            # Update the online mean value
            for cluster in range(num_clusters):
                new_samples = X[i:i+batch_size][nearest_centroid[i:i+batch_size] == cluster].shape[0]

                if new_samples > 0:
                    sum = X[i:i+batch_size][nearest_centroid[i:i+batch_size] == cluster].sum(dim=0)
                    n_samples[cluster] += new_samples
                    means[cluster] += (sum - new_samples*means[cluster])/n_samples[cluster]

        centroids = means
        center_shift = torch.norm(centroids-centroids_pre, dim=1).sum()
        iteration += 1

        pbar.set_postfix(
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}',
            device=f'{device}'
        )
        pbar.update(1)

        # Convergence check
        if center_shift ** 2 < tol:
            break

        if iter_limit != 0 and iteration >= iter_limit:
            break

    pbar.close()
    return nearest_centroid.cpu(), centroids.cpu()


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu'),
        gamma_for_soft_dtw=0.001
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :param gamma_for_soft_dtw: approaches to (hard) DTW as gamma -> 0
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = partial(pairwise_distance, device=device)
    elif distance == 'cosine':
        pairwise_distance_function = partial(pairwise_cosine, device=device)
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, device=torch.device('cpu')):
    # transfer to device
    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis