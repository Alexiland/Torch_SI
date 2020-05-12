import numpy as np
import torch
from tqdm import tqdm
import math

"""
Author: Chaojian Li
"""

def initialize(X, num_clusters):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    initial_state = X[indices]
    return initial_state


def kmeans(
        X,
        num_clusters,
        distance='euclidean',
        cluster_centers=[],
        tol=1e-4,
        device=torch.device('cpu'),
        verbose=True,
        max_iters=30
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if verbose:
        print(f'running k-means on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    # initialize
    if type(cluster_centers) == list:  # ToDo: make this less annoyingly weird
        initial_state = initialize(X, num_clusters)
    else:
        if verbose:
            print('resuming')
        # find data point closest to the initial cluster center
        initial_state = cluster_centers
        dis = pairwise_distance_function(X, initial_state, device)
        choice_points = torch.argmin(dis, dim=0)
        initial_state = X[choice_points]
        initial_state = initial_state.to(device)

    iteration = 0
    if verbose:
        tqdm_meter = tqdm(desc='[running kmeans]')
    while True:

        dis = pairwise_distance_function(X, initial_state, device)

        choice_cluster = torch.argmin(dis, dim=1).to(device)

        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze().to(device)

            selected = torch.index_select(X, 0, selected)

            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        # increment iteration
        iteration = iteration + 1

        if verbose:
            # update tqdm meter
            tqdm_meter.set_postfix(
                iteration=f'{iteration}',
                center_shift=f'{center_shift ** 2:0.6f}',
                tol=f'{tol:0.6f}'
            )
            tqdm_meter.update()
        if center_shift ** 2 < tol or iteration == max_iters:
            break

    return choice_cluster, initial_state


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers, device)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster


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

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    A_normalized = A / A.norm(dim=-1, keepdim=True)
    B_normalized = B / B.norm(dim=-1, keepdim=True)

    cosine = A_normalized * B_normalized

    # return N*N matrix for pairwise distance
    cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    return cosine_dis


def kmeans_cluster(x, max_clusters=3, max_iters=30):
    # return format: indices_list, mean_list
    # Reshape input for being processed by kmeans
    x = x.reshape((int(x.size()[0]), 1))
    for i in range(max_clusters):
        if max_clusters - i != 1:
            cluster_ids_x, cluster_centers = kmeans(
                X=x, num_clusters=max_clusters - i, device=torch.device('cuda:0'), verbose=False, max_iters=max_iters
            )
            cluster_ids_x = torch.squeeze(cluster_ids_x)
            mean_list = torch.squeeze(cluster_centers)
            mean_list = [value for value in mean_list] + [None] * i
            indices_list = []
            for v in range(max_clusters - i):
                indices_list.append((cluster_ids_x == v).nonzero().squeeze())
            indices_list = indices_list + [torch.tensor([], dtype=torch.int64).cuda()] * i
            if torch.sum(cluster_centers != cluster_centers):
                continue
            else:
                break
        else:
            # for one cluster, just return the mean
            indices_list = [torch.arange(0, int(x.size()[0])).cuda(), torch.tensor([], dtype=torch.int64).cuda(),
                            torch.tensor([], dtype=torch.int64).cuda()]
            mean_list = [torch.mean(x), None, None]
    return indices_list, mean_list


def sort_cluster(x, max_clusters=3):
    x = x.squeeze()
    assert int(x.size()[0]) > max_clusters, ("The # samples is less than the max_clusters")
    steps = int(math.ceil(int(x.size()[0]) / max_clusters))
    break_point_list = [steps * (i + 1) for i in range(max_clusters - 1)]
    sort_indexes = torch.argsort(x)
    indices_list = []
    mean_list = []
    prev_break_point = -1
    for break_point in break_point_list:
        indices_list.append(((sort_indexes >= prev_break_point) & (sort_indexes < break_point)).nonzero().squeeze())
        mean_list.append(torch.mean(torch.index_select(x, 0, indices_list[-1])))
        prev_break_point = break_point
    indices_list.append((sort_indexes >= break_point_list[-1]).nonzero().squeeze())
    mean_list.append(torch.mean(torch.index_select(x, 0, indices_list[-1])))
    return indices_list, mean_list