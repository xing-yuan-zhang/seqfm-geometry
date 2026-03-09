import os
import argparse
import numpy as np

def read_ids(path):
    xs = []
    with open(path, "r") as f:
        for line in f:
            t = line.strip()
            if t:
                xs.append(t)
    return xs

def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

def write_edges(path, edges):
    with open(path, "w") as f:
        for u, v, w in edges:
            f.write(u + "\t" + v + "\t" + f"{w:.6g}" + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--outdir", default="graphs")
    ap.add_argument("--cand_M", type=int, default=500)
    ap.add_argument("--topm", type=int, default=20)
    ap.add_argument("--weight", choices=["cos_pos", "row_softmax"], default="cos_pos")
    ap.add_argument("--tau", type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    z = np.load(args.emb_npz, allow_pickle=True)
    ids = list(z["ids"])
    emb = np.array(z["emb"], dtype=np.float32)
    id2i = {k:i for i,k in enumerate(ids)}

    nodes = read_ids(args.nodes)
    nodes = [x for x in nodes if x in id2i]
    X = l2norm(emb[[id2i[x] for x in nodes]]).astype(np.float32)

    G = X @ X.T
    np.fill_diagonal(G, -1.0)

    N = len(nodes)
    M = min(args.cand_M, N - 1)
    m = min(args.topm, M)

    raw = []
    for i in range(N):
        idx = np.argpartition(-G[i], M)[:M]
        sims = G[i, idx].astype(np.float64)

        if args.weight == "row_softmax":
            x = sims / max(args.tau, 1e-12)
            x = x - x.max()
            w = np.exp(x)
            w = w / (w.sum() + 1e-12)
        else:
            w = np.maximum(sims, 0.0)

        sel = np.argpartition(-w, m)[:m]
        u = nodes[i]
        for k in sel:
            j = int(idx[int(k)])
            v = nodes[j]
            ww = float(w[int(k)])
            raw.append((u, v, ww))

    dct = {}
    for u, v, w in raw:
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in dct or w > dct[key]:
            dct[key] = w

    out = [(u, v, w) for (u, v), w in dct.items() if w > 0]
    out.sort()
    out_path = os.path.join(args.outdir, f"fm_sim_knn_M{M}_m{m}_{args.weight}.tsv")
    write_edges(out_path, out)

    meta_path = os.path.join(args.outdir, f"fm_sim_knn_M{M}_m{m}_{args.weight}.meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"emb_npz\t{args.emb_npz}\n")
        f.write(f"nodes\t{args.nodes}\n")
        f.write(f"cand_M\t{M}\n")
        f.write(f"topm\t{m}\n")
        f.write(f"weight\t{args.weight}\n")
        f.write(f"tau\t{args.tau}\n")

if __name__ == "__main__":
    main()