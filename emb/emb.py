"""
Embedding STRING nodelist with ESM2 or ProtT5.

Output:
- A .npz file containing:
  - ids: an array of node IDs (dtype=object)
  - emb: an array of node embeddings (dtype=float32, shape=(n_nodes, embedding_dim))

Call:
    from emb import embed_main
    res = embed_main(
        backend="esm2",
        model="esm2_t33_650M_UR50D",
        fasta="nodes.fasta",
        node_ids="nodes.ids",
        out="embeddings.npz",
        max_len=1022,
        stride=768,
        bucket=128,
        win_bs=16,
        fp16=True,
        l2norm=True,
        cpu_threads=4,
        keep_npy=False,
    )
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm

def read_ids(path):
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]

def read_fasta(path):
    seqs = {}
    k, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if k is not None:
                    seqs[k] = "".join(buf)
                k = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if k is not None:
            seqs[k] = "".join(buf)
    return seqs

def l2norm_rows(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def slide_seq(s, win, stride):
    if len(s) <= win:
        return [s]
    out, i = [], 0
    while True:
        out.append(s[i:i+win])
        if i + win >= len(s):
            break
        i += stride
    return out

@torch.no_grad()
def embed_windows(model, tok, parts, device, max_len, bs, use_amp, seq_transform=None, drop_special=False):
    acc = None
    n = 0

    for i in range(0, len(parts), bs):
        b = parts[i:i+bs]
        if seq_transform is not None:
            b = [seq_transform(x) for x in b]

        x = tok(
            b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
        x = {k: v.to(device) for k, v in x.items()}

        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                h = model(**x).last_hidden_state
        else:
            h = model(**x).last_hidden_state

        attn = x["attention_mask"].bool()

        if drop_special:
            keep = attn.clone()
            keep[:, 0] = False
            lens = attn.long().sum(dim=1)
            last = torch.clamp(lens - 1, min=0)
            keep[torch.arange(keep.size(0), device=keep.device), last] = False
            m = keep.unsqueeze(-1).to(h.dtype)
            denom = m.sum(dim=1).clamp_min(1.0)
        else:
            m = attn.unsqueeze(-1).to(h.dtype)
            denom = m.sum(dim=1).clamp_min(1.0)

        v = (h * m).sum(dim=1) / denom
        s = v.sum(dim=0)

        acc = s if acc is None else (acc + s)
        n += v.size(0)

        del h, attn, m, denom, v, s, x
        if drop_special:
            del keep, lens, last
        if device == "cuda":
            torch.cuda.empty_cache()

    return acc / max(n, 1)

def build_model(backend, model_name, device):
    if backend == "esm2":
        tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AutoModel.from_pretrained(model_name).eval().to(device)
        D = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "d_model", 0))
        if D <= 0:
            raise RuntimeError("cannot infer embedding dim from model config")
        return tok, model, D

    if backend == "prott5":
        tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False)
        m = AutoModel.from_pretrained(model_name).eval().to(device)
        model = m.get_encoder()
        D = int(getattr(model.config, "d_model", 0) or getattr(model.config, "hidden_size", 0))
        if D <= 0:
            raise RuntimeError("cannot infer embedding dim from model config")
        return tok, model, D

    raise RuntimeError(f"unsupported backend: {backend}")

def embed_main(
    backend,
    model,
    fasta,
    node_ids,
    out,
    max_len=None,
    stride=768,
    bucket=128,
    win_bs=16,
    fp16=False,
    l2norm=False,
    cpu_threads=4,
    keep_npy=False,
):
    torch.set_num_threads(max(1, cpu_threads))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(fp16 and device == "cuda")

    if max_len is None:
        max_len = 1022 if backend == "esm2" else 1024

    fasta_path = Path(fasta)
    ids_path = Path(node_ids)
    out_npz = Path(out)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    tmp_npy = out_npz.with_suffix(".npy")

    seqs = read_fasta(fasta_path)
    ids = read_ids(ids_path)

    tok, model_obj, D = build_model(backend, model, device)

    dtype = np.float16 if (fp16 and device == "cuda") else np.float32
    mm = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=dtype, shape=(len(ids), D))

    items = []
    for nid in ids:
        s = seqs.get(nid)
        if s is None:
            raise RuntimeError(f"missing sequence: {nid}")
        items.append((nid, s, len(s)))
    items.sort(key=lambda x: x[2])

    buckets = {}
    for nid, s, L in items:
        k = (L // bucket) * bucket
        buckets.setdefault(k, []).append((nid, s))

    idx_map = {nid: i for i, nid in enumerate(ids)}
    pbar = tqdm(total=len(ids), desc=f"Embedding ({backend})")

    if backend == "esm2":
        seq_transform = None
        drop_special = True
    else:
        seq_transform = lambda x: " ".join(list(x))
        drop_special = False

    for k in sorted(buckets.keys()):
        for nid, s in buckets[k]:
            parts = slide_seq(s, max_len, stride)
            v = embed_windows(
                model_obj,
                tok,
                parts,
                device,
                max_len,
                win_bs,
                use_amp,
                seq_transform=seq_transform,
                drop_special=drop_special,
            )

            i = idx_map[nid]
            if dtype == np.float16:
                mm[i, :] = v.float().cpu().numpy().astype(np.float16, copy=False)
            else:
                mm[i, :] = v.float().cpu().numpy()

            if (pbar.n + 1) % 256 == 0:
                mm.flush()
            pbar.update(1)

    pbar.close()
    mm.flush()

    if l2norm:
        mm2 = np.lib.format.open_memmap(tmp_npy, mode="r+", dtype=dtype, shape=(len(ids), D))
        bs = 4096
        for i in range(0, len(ids), bs):
            x = np.array(mm2[i:i+bs], dtype=np.float32, copy=True)
            x = l2norm_rows(x)
            if dtype == np.float16:
                mm2[i:i+bs] = x.astype(np.float16, copy=False)
            else:
                mm2[i:i+bs] = x
        mm2.flush()

    emb = np.load(tmp_npy, mmap_mode=None)
    np.savez(out_npz, ids=np.array(ids, dtype=object), emb=emb.astype(np.float32))

    if not keep_npy:
        try:
            tmp_npy.unlink()
        except Exception:
            pass

    return {
        "backend": backend,
        "model": model,
        "device": device,
        "out": str(out_npz),
        "n_ids": len(ids),
        "dim": D,
    }