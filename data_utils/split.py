"""
Split edges into train/val/test sets for STRING supervised embed training.

Call:
    from emb_split import emb_split
    emb_split(
        emb_npz="esm2_t33_650m.npz",
        ppi_edges="string_edges.tsv",
        v_final="node_ids.txt",
        split="edge",
        outdir="splits_edge",
        dedup_undirected_flag=True,
    )
"""

import os
import numpy as np
from pathlib import Path

def read_ids(path):
    if not path:
        return None
    xs = []
    with open(path, "r") as f:
        for line in f:
            xs.append(line.strip())
    return set(xs)

def read_edges(path):
    edges = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = line.split("\t")
            if len(s) < 2 or s[0].startswith("node"):
                continue
            u, v = s[0], s[1]
            if u != v:
                edges.append((u, v))
    return edges

def write_nodes(path, xs):
    with open(path, "w") as f:
        for x in xs:
            f.write(str(x) + "\n")

def write_edges(path, edges):
    with open(path, "w") as f:
        for u, v in edges:
            f.write(u + "\t" + v + "\n")

def dedup_undirected(edges):
    seen = set()
    out = []
    for u, v in edges:
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((a, b))
    return out

def emb_split(
    emb_npz,
    ppi_edges,
    v_final,
    outdir=".",
    seed=0,
    train_frac=0.8,
    val_frac=0.1,
    split="edge",
    dedup_undirected_flag=False,
):
    base = Path(__file__).resolve().parent
    os.makedirs(base / outdir, exist_ok=True)

    z = np.load(base / emb_npz, allow_pickle=True)
    ids = list(z["ids"])
    idset = set(ids)

    v_final_set = read_ids(base / v_final)
    if v_final_set is None:
        v_final_set = idset
    else:
        v_final_set = v_final_set & idset

    edges = read_edges(base / ppi_edges)
    edges = [(u, v) for (u, v) in edges if (u in v_final_set and v in v_final_set)]
    if dedup_undirected_flag:
        edges = dedup_undirected(edges)

    rng = np.random.RandomState(seed)

    if split == "edge":
        rng.shuffle(edges)
        m = len(edges)
        m_train = int(m * train_frac)
        m_val = int(m * val_frac)

        trE = edges[:m_train]
        vaE = edges[m_train:m_train + m_val]
        teE = edges[m_train + m_val:]

        train_nodes = sorted(set([u for u, v in trE] + [v for u, v in trE]))
        val_nodes = sorted(set([u for u, v in vaE] + [v for u, v in vaE]))
        test_nodes = sorted(set([u for u, v in teE] + [v for u, v in teE]))

        meta = {
            "split": "edge",
            "n_nodes_total": len(v_final_set),
            "n_edges_total": m,
            "n_train_pos": len(trE),
            "n_val_pos": len(vaE),
            "n_test_pos": len(teE),
            "n_train_nodes": len(train_nodes),
            "n_val_nodes": len(val_nodes),
            "n_test_nodes": len(test_nodes),
            "seed": seed,
            "dedup_undirected": int(dedup_undirected_flag),
        }

    else:
        nodes = sorted(set([u for u, _ in edges] + [v for _, v in edges]))
        rng.shuffle(nodes)

        n = len(nodes)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_nodes = set(nodes[:n_train])
        val_nodes = set(nodes[n_train:n_train + n_val])
        test_nodes = set(nodes[n_train + n_val:])

        def filt_edges(node_set):
            out = []
            for u, v in edges:
                if u in node_set and v in node_set:
                    out.append((u, v))
            return out

        trE = filt_edges(train_nodes)
        vaE = filt_edges(val_nodes)
        teE = filt_edges(test_nodes)

        train_nodes = sorted(train_nodes)
        val_nodes = sorted(val_nodes)
        test_nodes = sorted(test_nodes)

        meta = {
            "split": "node",
            "n_nodes_total": len(v_final_set),
            "n_nodes_with_edges": n,
            "n_train_nodes": len(train_nodes),
            "n_val_nodes": len(val_nodes),
            "n_test_nodes": len(test_nodes),
            "n_train_pos": len(trE),
            "n_val_pos": len(vaE),
            "n_test_pos": len(teE),
            "seed": seed,
            "dedup_undirected": int(dedup_undirected_flag),
        }

    write_nodes(base / outdir / "train_nodes.txt", train_nodes)
    write_nodes(base / outdir / "val_nodes.txt", val_nodes)
    write_nodes(base / outdir / "test_nodes.txt", test_nodes)

    write_edges(base / outdir / "train_pos_edges.tsv", trE)
    write_edges(base / outdir / "val_pos_edges.tsv", vaE)
    write_edges(base / outdir / "test_pos_edges.tsv", teE)

    with open(base / outdir / "meta.txt", "w") as f:
        for k in sorted(meta.keys()):
            f.write(f"{k}\t{meta[k]}\n")

    return {
        "train_nodes": train_nodes,
        "val_nodes": val_nodes,
        "test_nodes": test_nodes,
        "train_pos_edges": trE,
        "val_pos_edges": vaE,
        "test_pos_edges": teE,
        "meta": meta,
    }