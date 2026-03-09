import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def read_ids(path):
    xs = []
    with open(path, "r") as f:
        for line in f:
            t = line.strip()
            if t:
                xs.append(t)
    return xs


def read_edges(path):
    es = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip().split("\t")
            if len(s) < 2:
                continue
            u, v = s[0], s[1]
            if u == v:
                continue
            es.append((u, v))
    return es


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


class PairMLP(nn.Module):
    def __init__(self, d, hidden, drop):
        super().__init__()
        self.fc1 = nn.Linear(d * 4, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = drop

    def forward(self, eu, ev):
        x = torch.cat([eu, ev, torch.abs(eu - ev), eu * ev], dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.out(x).squeeze(1)
        return x


def neg_sample_degree_matched(pos_edges, nodes, deg, n_neg, rng):
    nodes_arr = np.array(nodes, dtype=object)
    w = np.array([deg.get(x, 1) for x in nodes_arr], dtype=np.float64)
    w = w / (w.sum() + 1e-12)

    pos_set = set()
    for u, v in pos_edges:
        if u < v:
            pos_set.add((u, v))
        else:
            pos_set.add((v, u))

    neg = []
    neg_set = set()
    tries = 0
    max_tries = max(n_neg * 50, 1000)

    while len(neg) < n_neg and tries < max_tries:
        u = rng.choice(nodes_arr, p=w)
        v = rng.choice(nodes_arr, p=w)
        tries += 1

        if u == v:
            continue

        a, b = (u, v) if u < v else (v, u)

        if (a, b) in pos_set:
            continue
        if (a, b) in neg_set:
            continue

        neg.append((a, b))
        neg_set.add((a, b))

    return neg


def batched_logits(model, E, pairs, bs, device):
    if len(pairs) == 0:
        return np.array([], dtype=np.float32)

    us = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    eu = E[us]
    ev = E[vs]

    out = []
    n = len(pairs)
    i = 0
    while i < n:
        j = min(n, i + bs)
        lu = torch.from_numpy(eu[i:j]).to(device)
        lv = torch.from_numpy(ev[i:j]).to(device)
        with torch.no_grad():
            z = model(lu, lv).detach().cpu().numpy()
        out.append(z)
        i = j

    return np.concatenate(out, axis=0)


def pair_mlp(
    emb_npz,
    train_pos,
    val_pos,
    nodes,
    outdir="models",
    graph_outdir="graphs",
    seed=0,
    device="cuda",
    hidden=512,
    drop=0.1,
    lr=1e-3,
    wd=1e-4,
    epochs=5,
    batch=4096,
    neg_ratio=1.0,
    cand_M=500,
    topm=20,
    score_to_prob="sigmoid",
):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(graph_outdir, exist_ok=True)

    rng = np.random.RandomState(seed)

    z = np.load(emb_npz, allow_pickle=True)
    ids = list(z["ids"])
    emb = np.array(z["emb"], dtype=np.float32)
    id2i = {k: i for i, k in enumerate(ids)}

    nodes_list = read_ids(nodes)
    nodes_list = [x for x in nodes_list if x in id2i]
    if len(nodes_list) < 2:
        raise ValueError("fewer than 2 valid nodes found in emb_npz")

    emb_nodes = l2norm(emb[[id2i[x] for x in nodes_list]])
    E = {nodes_list[i]: emb_nodes[i] for i in range(len(nodes_list))}

    tr_pos = read_edges(train_pos)
    va_pos = read_edges(val_pos)

    tr_pos = [(u, v) for (u, v) in tr_pos if u in E and v in E]
    va_pos = [(u, v) for (u, v) in va_pos if u in E and v in E]

    if len(tr_pos) == 0:
        raise ValueError("no valid training positive edges after filtering")
    if len(va_pos) == 0:
        raise ValueError("no valid validation positive edges after filtering")

    deg = {}
    for u, v in tr_pos:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    n_tr_neg = int(len(tr_pos) * neg_ratio)
    n_va_neg = int(len(va_pos) * neg_ratio)

    tr_neg = neg_sample_degree_matched(tr_pos, nodes_list, deg, n_tr_neg, rng)
    va_neg = neg_sample_degree_matched(va_pos, nodes_list, deg, n_va_neg, rng)

    def build_xy(pos, neg):
        pairs = pos + neg
        y = np.concatenate([
            np.ones(len(pos), dtype=np.float32),
            np.zeros(len(neg), dtype=np.float32),
        ])
        idx = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in idx]
        y = y[idx]
        return pairs, y

    tr_pairs, tr_y = build_xy(tr_pos, tr_neg)
    va_pairs, va_y = build_xy(va_pos, va_neg)

    d = emb_nodes.shape[1]

    if device == "cpu":
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = PairMLP(d, hidden, drop).to(torch_device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    E_mat = {k: np.asarray(v, dtype=np.float32) for k, v in E.items()}

    def iter_batches(pairs, y, bs):
        n = len(pairs)
        for i in range(0, n, bs):
            j = min(n, i + bs)
            ps = pairs[i:j]
            yy = torch.from_numpy(y[i:j]).to(torch_device)
            eu = torch.from_numpy(np.stack([E_mat[u] for u, _ in ps], axis=0)).to(torch_device)
            ev = torch.from_numpy(np.stack([E_mat[v] for _, v in ps], axis=0)).to(torch_device)
            yield eu, ev, yy

    best = 1e18
    ckpt_path = os.path.join(outdir, "pair_mlp.pt")

    for ep in range(epochs):
        model.train()
        for eu, ev, yy in iter_batches(tr_pairs, tr_y, batch):
            opt.zero_grad(set_to_none=True)
            z = model(eu, ev)
            loss = F.binary_cross_entropy_with_logits(z, yy)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            losses = []
            for eu, ev, yy in iter_batches(va_pairs, va_y, batch):
                z = model(eu, ev)
                losses.append(F.binary_cross_entropy_with_logits(z, yy).item())
            vloss = float(np.mean(losses)) if losses else 1e18

        if vloss < best:
            best = vloss
            torch.save({"state_dict": model.state_dict(), "d": d}, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=torch_device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X = emb_nodes.astype(np.float32)
    G = X @ X.T
    np.fill_diagonal(G, -1.0)

    N = len(nodes_list)
    M = min(cand_M, N - 1)
    topm = min(topm, M)

    edges = []
    for i in range(N):
        idx = np.argpartition(-G[i], M)[:M]
        cand = [nodes_list[j] for j in idx]
        u = nodes_list[i]
        pairs = [(u, v) for v in cand]

        logits = batched_logits(model, E_mat, pairs, bs=8192, device=torch_device)

        if score_to_prob == "sigmoid":
            scores = 1.0 / (1.0 + np.exp(-logits))
        else:
            scores = logits

        sel = np.argpartition(-scores, topm)[:topm]
        for k in sel:
            v = cand[k]
            w = float(scores[k])
            edges.append((u, v, w))

    dct = {}
    for u, v, w in edges:
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in dct or w > dct[key]:
            dct[key] = w

    out_path = os.path.join(graph_outdir, f"fmppi_mlp_M{M}_m{topm}.tsv")
    with open(out_path, "w") as f:
        for (u, v), w in sorted(dct.items()):
            f.write(u + "\t" + v + "\t" + f"{w:.6g}" + "\n")

    meta_path = os.path.join(graph_outdir, f"fmppi_mlp_M{M}_m{topm}.meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"emb_npz\t{emb_npz}\n")
        f.write(f"train_pos\t{train_pos}\n")
        f.write(f"val_pos\t{val_pos}\n")
        f.write(f"nodes\t{nodes}\n")
        f.write(f"hidden\t{hidden}\n")
        f.write(f"drop\t{drop}\n")
        f.write(f"lr\t{lr}\n")
        f.write(f"wd\t{wd}\n")
        f.write(f"epochs\t{epochs}\n")
        f.write(f"batch\t{batch}\n")
        f.write(f"neg_ratio\t{neg_ratio}\n")
        f.write(f"cand_M\t{M}\n")
        f.write(f"topm\t{topm}\n")
        f.write(f"score_to_prob\t{score_to_prob}\n")
        f.write(f"val_bce\t{best}\n")
        f.write(f"num_nodes\t{N}\n")
        f.write(f"num_train_pos\t{len(tr_pos)}\n")
        f.write(f"num_val_pos\t{len(va_pos)}\n")
        f.write(f"num_train_neg\t{len(tr_neg)}\n")
        f.write(f"num_val_neg\t{len(va_neg)}\n")

    return {
        "model_ckpt": ckpt_path,
        "graph_tsv": out_path,
        "meta_txt": meta_path,
        "val_bce": best,
        "num_nodes": N,
        "cand_M": M,
        "topm": topm,
    }
