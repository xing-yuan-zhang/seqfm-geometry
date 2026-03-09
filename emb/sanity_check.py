"""
Sanity check for ESM2 or ProtT5 embeddings:
- Check for NaN/Inf values in a random sample of rows.
- Check the distribution of row norms and values.
- Check the distribution of top cosine similarities among random samples.
- Check the correlation between sequence length and embedding norm.

Call:
    from sanity_check import sanity_check
    sanity_check("embeddings.npz", "nodes.length")
"""

import numpy as np

def _rand_idx(n, k, seed=0):
    rng = np.random.default_rng(seed)
    k = min(k, n)
    return rng.choice(n, size=k, replace=False)

def basic(path, k=2048, seed=0):
    emb = np.load(path, mmap_mode="r")
    print("shape", emb.shape, "dtype", emb.dtype)

    n, d = emb.shape
    idx = _rand_idx(n, k, seed=seed)
    x = np.array(emb[idx], dtype=np.float32, copy=True)

    if not np.isfinite(x).all():
        bad = np.where(~np.isfinite(x))
        print("NaN/Inf found in sampled rows:", bad[0][:10])
        return False

    row_norm = np.linalg.norm(x, axis=1)
    print("row_norm quantiles:",
          np.quantile(row_norm, [0, 0.01, 0.5, 0.99, 1.0]))

    flat = x.reshape(-1)
    print("value quantiles:",
          np.quantile(flat, [0, 0.01, 0.5, 0.99, 1.0]))
    print("min/max:", float(flat.min()), float(flat.max()))

    zeroish = np.mean(row_norm < 1e-6)
    if zeroish > 0:
        print("warning: fraction near-zero rows:", float(zeroish))

    return True

def finite_stream(path, chunk=4096):
    emb = np.load(path, mmap_mode="r")
    n = emb.shape[0]
    for i in range(0, n, chunk):
        x = np.array(emb[i:i+chunk], dtype=np.float32, copy=True)
        if not np.isfinite(x).all():
            print("NaN/Inf found in chunk:", i, i+chunk)
            return False
    print("finite check: OK (stream)")
    return True

def knn_cosine(path, k=512, top=5, seed=0):
    emb = np.load(path, mmap_mode="r")
    n, d = emb.shape
    idx = _rand_idx(n, k, seed=seed)
    x = np.array(emb[idx], dtype=np.float32, copy=True)

    xn = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    x = x / xn

    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)

    top1 = np.max(sim, axis=1)
    top5 = np.sort(sim, axis=1)[:, -top]
    rand = sim[np.triu_indices(k, 1)]

    print("cos top1 quantiles:", np.quantile(top1, [0.01, 0.5, 0.99]))
    print("cos top%d-min quantiles:" % top, np.quantile(top5, [0.01, 0.5, 0.99]))
    print("cos random quantiles:", np.quantile(rand, [0.01, 0.5, 0.99]))

    return True

def length_correlation(path, lengths, k=10000, seed=0):
    emb = np.load(path, mmap_mode="r")
    n = emb.shape[0]
    if len(lengths) != n:
        print("length mismatch:", n, "!=", len(lengths))
        return False

    idx = _rand_idx(n, min(k, n), seed=seed)
    x = np.array(emb[idx], dtype=np.float32, copy=True)
    L = np.asarray(lengths, dtype=np.float32)[idx]

    r = np.linalg.norm(x, axis=1)

    def rank(a):
        o = np.argsort(a)
        r = np.empty_like(o, dtype=np.float32)
        r[o] = np.arange(len(a), dtype=np.float32)
        return r

    Lr = rank(L)
    rr = rank(r)
    corr = np.corrcoef(Lr, rr)[0, 1]
    print("spearman corr(length, norm) ~", float(corr))
    return True

def sanity_check(npy_path: str, seqs_length_path: str):
    basic(npy_path)
    knn_cosine(npy_path)
    length = [float(x.strip()) for x in open(seqs_length_path)]
    length_correlation(npy_path, length)