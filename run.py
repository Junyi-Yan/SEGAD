import argparse
import random
import time
import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
from model import Model
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="SEGAD")
    parser.add_argument("--dataset",type=str,default="YelpChi",help="Dataset name (photo|reddit|Amazon|elliptic|tf_finace|YelpChi).",)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, default=[0, 1, 2])
    parser.add_argument("--embedding_dim", type=int, default=300)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--drop_prob", type=float, default=0.0)
    parser.add_argument("--readout", type=str, default="avg")
    parser.add_argument("--auc_test_rounds", type=int, default=256)
    parser.add_argument("--alpha",type=float,default=0.4,help="Weight of the anomaly neighborhood-difference loss.")
    parser.add_argument("--beta",type=float,default=0.8,help="Weight of the inter-class affinity loss.")
    parser.add_argument("--negsamp_ratio", type=int, default=1)
    parser.add_argument("--mean", type=float, default=0.0)
    parser.add_argument("--var", type=float, default=0.0)
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--init_ratio", type=float, default=0.001)
    parser.add_argument("--epo_label", type=int, default=10)
    parser.add_argument("--init_num", type=int, default=1000)
    return parser.parse_args()


def set_random_seed(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data(dataset):
    adj,features,_,idx_train,_,idx_test,ano_label,_,_,= load_mat(dataset)
    if dataset in ["Amazon", "tf_finace", "reddit", "elliptic", "ogbn_arixv"]:
        features, _ = preprocess_features(features)
    else:
        features = features.todense()

    raw_adj = (adj + sp.eye(adj.shape[0])).todense()
    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()

    features = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
    adj = torch.as_tensor(adj, dtype=torch.float32).unsqueeze(0)

    return features,adj,idx_train,idx_test,ano_label,


def initialize_labels(features, adj, idx_train, ano_label, num_nodes, args):
    all_normal_idx = [idx for idx in idx_train if ano_label[idx] == 0]
    all_abnormal_idx = [idx for idx in idx_train if ano_label[idx] == 1]

    sample_size = max(1, int(num_nodes * args.init_ratio))
    update_limit = int(num_nodes * (args.ratio - 2 * args.init_ratio))

    if args.init_num > len(idx_train):
        raise ValueError(f"init_num ({args.init_num}) exceeds the training-set size ({len(idx_train)}).")

    initial_pool = random.sample(list(idx_train), args.init_num)
    pool_features = features[0, initial_pool]
    pool_adj = adj[0, initial_pool, :][:, initial_pool]
    normal_idx, abnormal_idx = compute_cosine_similarity(pool_features,pool_adj,initial_pool,sample_size,ano_label)
    labeled_idx = list(normal_idx) + list(abnormal_idx)
    available_idx = list(set(idx_train) - set(labeled_idx))

    return {"all_normal_idx": all_normal_idx,
            "all_abnormal_idx": all_abnormal_idx,
            "normal_idx": list(normal_idx),
            "abnormal_idx": list(abnormal_idx),
            "labeled_idx": labeled_idx,
            "available_idx": available_idx,
            "update_limit": update_limit}


def update_labels(embeddings, state, stage_weight, total_weight):
    dbscan = DBSCAN(eps=0.001, min_samples=2, metric="cosine")
    cluster_labels = dbscan.fit_predict(embeddings)
    outlier_idx = np.where(cluster_labels == -1)[0]
    train_outlier_idx = np.intersect1d(outlier_idx, state["available_idx"])
    ranked_idx, _ = compute_average_cosine_similarity(embeddings,train_outlier_idx,state["normal_idx"],state["abnormal_idx"],)
    stage_size = round(state["update_limit"] * stage_weight / total_weight)
    queried_idx = ranked_idx[:stage_size]
    state["abnormal_idx"] += np.intersect1d(queried_idx, state["all_abnormal_idx"]).tolist()
    state["normal_idx"] += np.intersect1d(queried_idx, state["all_normal_idx"]).tolist()
    state["labeled_idx"] = state["abnormal_idx"] + state["normal_idx"]
    queried_set = set(queried_idx)
    state["available_idx"] = [idx for idx in state["available_idx"] if idx not in queried_set]


def run_single_seed(args, seed):
    print(f"Dataset: {args.dataset}; seed: {seed}")
    set_random_seed(seed)
    features, adj, idx_train, idx_test, ano_label = prepare_data(args.dataset)
    num_nodes = features.shape[1]
    feature_dim = features.shape[2]

    model = Model(feature_dim,args.embedding_dim,"prelu",args.negsamp_ratio,args.readout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    features = features.to(device)
    adj = adj.to(device)

    pos_weight = torch.tensor([args.negsamp_ratio], device=device)
    bce_loss = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    state = initialize_labels(features, adj, idx_train, ano_label, num_nodes, args)
    num_stages = round(args.num_epoch / args.epo_label)
    stage_weight = num_stages
    total_weight = sum(np.arange(1, num_stages))
    total_time = 0.0
    test_auc = 0.0
    test_ap = 0.0

    with tqdm(total=args.num_epoch, desc="Training") as progress:
        for epoch in range(args.num_epoch):
            start_time = time.time()
            model.train()
            optimizer.zero_grad()

            abnormal_idx = state["abnormal_idx"]
            normal_idx = state["normal_idx"]
            outputs = model(features, adj, abnormal_idx, normal_idx, True, args)
            emb, _, logits, emb_con, emb_abnormal, _, emb_normal = outputs

            target = torch.cat((torch.zeros(len(normal_idx)), torch.ones(len(emb_con)))).view(1, -1, 1).to(device)
            if epoch % args.epo_label == 0:
                stage_weight -= 1
                detached_emb = emb.squeeze(0).detach().cpu()
                update_labels(detached_emb, state, stage_weight, total_weight)
                abnormal_idx = state["abnormal_idx"]
                normal_idx = state["normal_idx"]

            loss_bce = bce_loss(logits, target).mean()
            loss_ano = torch.sum((emb_con - emb_abnormal) ** 2, dim=1).mean()
            loss_aff = max_cosine_similarity_loss(emb_normal.squeeze(0), emb_abnormal.squeeze(0))
            loss = loss_bce + args.alpha * loss_ano + args.beta * loss_aff

            loss.backward()
            optimizer.step()
            total_time += time.time() - start_time

            model.eval()
            with torch.no_grad():
                outputs = model(features, adj, abnormal_idx, normal_idx, False, args)
                test_logits = outputs[2][:, idx_test, :].squeeze().cpu().numpy()

            test_auc = roc_auc_score(ano_label[idx_test], test_logits)
            test_ap = average_precision_score(ano_label[idx_test], test_logits)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:04d} | loss={loss.item():.5f} | "
                    f"AUROC={test_auc:.4f} | AUPRC={test_ap:.4f}")
            progress.update(1)

    labeled_ratio = len(state["labeled_idx"]) / num_nodes
    print(f"Labeled nodes: {len(state['labeled_idx'])} ({labeled_ratio:.2%}); "
        f"training time: {total_time:.2f}s")
    return test_auc, test_ap


def main():
    args = parse_args()
    auc_scores = []
    ap_scores = []
    for seed in args.seeds:
        auc, ap = run_single_seed(args, seed)
        auc_scores.append(auc)
        ap_scores.append(ap)
    print(f"AUROC: {np.mean(auc_scores) * 100:.2f} ± {np.std(auc_scores) * 100:.2f}")
    print(f"AUPRC: {np.mean(ap_scores) * 100:.2f} ± {np.std(ap_scores) * 100:.2f}")


if __name__ == "__main__":
    main()