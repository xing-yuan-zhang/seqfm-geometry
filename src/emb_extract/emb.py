import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

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
    out = []
    i = 0
    while True:
        out.append(s[i:i+win])
        if i + win >= len(s):
            break
        i += stride
    return out

@torch.no_grad()
def embed_windows(model, tok, parts, device, max_len, bs):
    out_vecs = []
    for i in range(0, len(parts), bs):
        b = parts[i:i+bs]
        x = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=max_len, add_special_tokens=True)
        x = {k:v.to(device) for k,v in x.items()}
        h = model(**x).last_hidden_state          # [B, L, H]
        attn = x["attention_mask"].bool()         # [B, L]

        keep = torch.zeros_like(attn)
        for j in range(attn.size(0)):
            idx = torch.where(attn[j])[0]
            if idx.numel() >= 3:
                keep[j, idx[1:-1]] = True
            else:
                keep[j, idx] = True

        m = keep.unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)   # [B, H]
        out_vecs.append(v)

    V = torch.cat(out_vecs, dim=0)               # [nwin, H]
    return V.mean(dim=0)                         # [H]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--node_ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--win_bs", type=int, default=64)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--l2norm", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = read_fasta(args.fasta)
    ids = read_ids(args.node_ids)

    tok = AutoTokenizer.from_pretrained(args.model, do_lower_case=False)
    model = AutoModel.from_pretrained(args.model).eval().to(device)

    is_t5 = "prot_t5" in args.model.lower() or "t5" in args.model.lower()
    if is_t5:
        raise RuntimeError("This script is ESM-like only (windowed + bucketing).")

    items = []
    miss = 0
    for nid in ids:
        s = seqs.get(nid)
        if s is None:
            miss += 1
        else:
            items.append((nid, s, len(s)))

    if miss:
        raise RuntimeError(f"missing sequences: {miss}/{len(ids)}")

    items.sort(key=lambda x: x[2])
    buckets = {}
    for nid, s, L in items:
        k = (L // args.bucket) * args.bucket
        buckets.setdefault(k, []).append((nid, s))

    embs = [None] * len(ids)
    idx_map = {nid:i for i, nid in enumerate(ids)}

    with torch.no_grad():
        for k in sorted(buckets.keys()):
            group = buckets[k]
            for nid, s in group:
                parts = slide_seq(s, args.max_len, args.stride)
                v = embed_windows(model, tok, parts, device, args.max_len, args.win_bs)
                if args.fp16 and device == "cuda":
                    v = v.half()
                embs[idx_map[nid]] = v.cpu()

    E = torch.stack(embs, dim=0).numpy()
    if args.l2norm:
        E = l2norm_rows(E.astype(np.float32)).astype(E.dtype, copy=False)
    np.save(args.out, E)

if __name__ == "__main__":
    main()
