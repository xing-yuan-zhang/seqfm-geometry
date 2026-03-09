import os
import json
import numpy as np
import networkx as nx

from io import read_nodes, load_edges_tsv, load_labels, load_seeds
from metrics import pr_auc, ranking_metrics

def eval_one_graph(
    edges_path,
    seeds_path,
    labels_path,
    out_prefix,
    nodes_whitelist_path="",
    alpha=0.85,
    min_w=0.0,
    topk=200,
    max_iter=500,
    tol=1e-10,
):
    node_whitelist = None
    if nodes_whitelist_path.strip():
        node_whitelist = read_nodes(nodes_whitelist_path)

    G = load_edges_tsv(edges_path, node_whitelist=node_whitelist, min_w=min_w)
    V = set(G.nodes())
    if node_whitelist is not None:
        V = V & node_whitelist

    pers = load_seeds(seeds_path, nodes_universe=V)
    seed_set = set(pers.keys())

    lab = load_labels(labels_path, nodes_universe=V)
    pos_all = set([n for n, v in lab.items() if v > 0])
    pos_eval = pos_all - seed_set

    scores = nx.pagerank(
        G,
        alpha=alpha,
        personalization=pers,
        weight="weight",
        max_iter=max_iter,
        tol=tol,
    )

    cand = [n for n in V if n not in seed_set]
    s = np.array([scores.get(n, 0.0) for n in cand], dtype=np.float64)
    y = np.array([1.0 if n in pos_eval else 0.0 for n in cand], dtype=np.float64)

    auc = pr_auc(s, y)

    idx = np.argsort(-s)
    top = [cand[i] for i in idx[:topk]]

    rank_res = ranking_metrics(top, pos_eval, cand, topk)

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    with open(f"{out_prefix}.top{topk}.txt", "w") as f:
        for n in top:
            f.write(n + "\n")

    res = {
        "edges": edges_path,
        "seeds": seeds_path,
        "labels": labels_path,
        "alpha": alpha,
        "topk": topk,
        "n_nodes": int(len(V)),
        "n_seeds_in_graph": int(len(seed_set)),
        "n_pos_total": int(len(pos_all)),
        "n_pos_eval": int(len(pos_eval)),
        "auprc": auc,
        "recall_at_k": rank_res["recall_at_k"],
        "precision_at_k": rank_res["precision_at_k"],
        "enrichment_factor_at_k": rank_res["enrichment_factor_at_k"],
    }

    with open(f"{out_prefix}.summary.json", "w") as f:
        json.dump(res, f, indent=2)

    return res