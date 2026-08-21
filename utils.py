from collections import Counter
import random

import dgl
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def sparse_to_tuple(sparse_mx, insert_batch=False):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack(
                (np.zeros(mx.row.shape[0]), mx.row, mx.col)
            ).transpose()
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            shape = mx.shape
        return coords, mx.data, shape

    if isinstance(sparse_mx, list):
        return [to_tuple(mx) for mx in sparse_mx]
    return to_tuple(sparse_mx)


def preprocess_features(features):
    """Row-normalize a sparse feature matrix."""
    rowsum = np.asarray(features.sum(1)).reshape(-1)
    with np.errstate(divide="ignore"):
        r_inv = np.power(rowsum, -1)
    r_inv[~np.isfinite(r_inv)] = 0.0
    normalized = sp.diags(r_inv).dot(features)
    return normalized.todense(), sparse_to_tuple(normalized)


def normalize_adj(adj):
    """Symmetrically normalize an adjacency matrix as D^-1/2 A D^-1/2."""
    adj = sp.coo_matrix(adj)
    rowsum = np.asarray(adj.sum(1)).reshape(-1)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def load_mat(dataset, train_rate=0.3, val_rate=0, data_dir="./dataset"):
    data = sio.loadmat(f"{data_dir}/{dataset}.mat")
    label = data["Label"] if "Label" in data else data["gnd"]
    attr = data["Attributes"] if "Attributes" in data else data["X"]
    network = data["Network"] if "Network" in data else data["A"]

    adj = sp.csr_matrix(network)
    features = sp.lil_matrix(attr)
    ano_labels = np.asarray(label).squeeze()

    if "str_anomaly_label" in data:
        str_ano_labels = np.asarray(data["str_anomaly_label"]).squeeze()
        attr_ano_labels = np.asarray(data["attr_anomaly_label"]).squeeze()
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_nodes = adj.shape[0]
    num_train = int(num_nodes * train_rate)
    num_val = int(num_nodes * val_rate)

    all_idx = list(range(num_nodes))
    random.shuffle(all_idx)
    idx_train = all_idx[:num_train]
    idx_val = all_idx[num_train:num_train + num_val]
    idx_test = all_idx[num_train + num_val:]

    print("Training", Counter(ano_labels[idx_train]))
    print("Validation", Counter(ano_labels[idx_val]))
    print("Test", Counter(ano_labels[idx_test]))

    return adj,features,all_idx,idx_train,idx_val,idx_test,ano_labels,str_ano_labels,attr_ano_labels



def adj_to_dgl_graph(adj):
    """Convert a SciPy adjacency matrix to a DGL graph."""
    src, dst = sp.coo_matrix(adj).nonzero()
    return dgl.graph((src, dst), num_nodes=adj.shape[0])


def inter_class_affinity_loss(normal_embeddings, abnormal_embeddings, power=3.0):
    if normal_embeddings.numel() == 0 or abnormal_embeddings.numel() == 0:
        raise ValueError()

    normal_embeddings = F.normalize(normal_embeddings, p=2, dim=-1)
    abnormal_embeddings = F.normalize(abnormal_embeddings, p=2, dim=-1)
    cosine_sim = torch.matmul(normal_embeddings, abnormal_embeddings.T)
    return (1.0 + cosine_sim).pow(power).mean()


def max_cosine_similarity_loss(x, y, alpha=3.0):
    """Backward-compatible wrapper for the inter-class affinity loss."""
    return inter_class_affinity_loss(x, y, power=alpha)


def compute_average_cosine_similarity(
    embeddings,
    outlier_indices,
    normal_label_idx,
    abnormal_label_idx,
):

    outlier_indices = np.asarray(outlier_indices, dtype=np.int64)
    if outlier_indices.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    if len(normal_label_idx) == 0 or len(abnormal_label_idx) == 0:
        raise ValueError(
            "Both normal and abnormal labelled nodes are required for "
            "affinity evaluation."
        )

    normal_embeddings = embeddings[normal_label_idx]
    abnormal_embeddings = embeddings[abnormal_label_idx]
    outlier_embeddings = embeddings[outlier_indices]

    outlier_embeddings = F.normalize(outlier_embeddings, p=2, dim=-1)
    normal_embeddings = F.normalize(normal_embeddings, p=2, dim=-1)
    abnormal_embeddings = F.normalize(abnormal_embeddings, p=2, dim=-1)
    affinities = (
        torch.matmul(outlier_embeddings, normal_embeddings.T).mean(dim=1)
        + torch.matmul(outlier_embeddings, abnormal_embeddings.T).mean(dim=1)
    )
    affinities = affinities.detach().cpu().numpy()

    order = np.argsort(affinities, kind="stable")
    return outlier_indices[order], affinities[order]


def random_walk_networkx(adj_matrix, start_node, walk_length=6):
    """Perform an unweighted random walk on an induced adjacency matrix."""
    if torch.is_tensor(adj_matrix):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    elif sp.issparse(adj_matrix):
        adj_matrix = adj_matrix.toarray()
    else:
        adj_matrix = np.asarray(adj_matrix)

    graph = nx.from_numpy_array(adj_matrix)
    walk = []
    current_node = int(start_node)

    for _ in range(walk_length):
        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break
        current_node = random.choice(neighbors)
        walk.append(current_node)

    return walk


def compute_cosine_similarity(features,adj,init_idx,number, ano_label,walk_length=6):
    """Select low-affinity normal and abnormal nodes for initialization."""
    init_idx = np.asarray(init_idx, dtype=np.int64)
    affinities = np.empty(len(init_idx), dtype=np.float64)

    for local_start in range(len(init_idx)):
        walk_nodes = random_walk_networkx(
            adj,
            local_start,
            walk_length=walk_length,
        )
        if not walk_nodes:
            affinities[local_start] = 1.0
            continue

        start_feature = features[local_start].unsqueeze(0)
        similarities = [
            F.cosine_similarity(
                start_feature,
                features[node].unsqueeze(0),
                dim=1,
            ).item()
            for node in walk_nodes
        ]
        affinities[local_start] = np.mean(similarities)

    order = np.argsort(affinities, kind="stable")
    sorted_global_idx = init_idx[order]
    sorted_labels = np.asarray(ano_label)[sorted_global_idx]

    normal_idx = sorted_global_idx[sorted_labels == 0][:number].tolist()
    abnormal_idx = sorted_global_idx[sorted_labels == 1][:number].tolist()
    return normal_idx, abnormal_idx


def select_random_points(idx_train, number, ano_label):
    """Randomly select up to ``number`` normal and abnormal training nodes."""
    normal_idx = [idx for idx in idx_train if ano_label[idx] == 0]
    abnormal_idx = [idx for idx in idx_train if ano_label[idx] == 1]
    random.shuffle(normal_idx)
    random.shuffle(abnormal_idx)
    return normal_idx[:number], abnormal_idx[:number]