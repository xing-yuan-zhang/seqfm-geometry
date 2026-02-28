import argparse
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
def embed_windows(model, tok, parts, device, max_len, bs, use_amp):
    acc = None
    n = 0
    for i in range(0, len(parts), bs):
        b = parts[i:i+bs]
        b = [" ".join(list(x)) for x in b]

        x = tok(
            b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True
        )
        x = {k: v.to(device) for k, v in x.items()}

        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                h = model(**x).last_hidden_state
        else:
            h = model(**x).last_hidden_state

        attn = x["attention_mask"].bool()
        m = attn.unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        s = v.sum(dim=0)
        acc = s if acc is None else (acc + s)
        n += v.shape[0]

        del h, attn, m, v, s, x
        if device == "cuda":
            torch.cuda.empty_cache()

    return acc / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--node_ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--win_bs", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--l2norm", action="store_true")
    ap.add_argument("--cpu_threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.cpu_threads))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.fp16 and device == "cuda")

    base = Path(__file__).resolve().parent
    seqs = read_fasta(base / args.fasta)
    ids = read_ids(base / args.node_ids)

    tok = AutoTokenizer.from_pretrained(args.model, do_lower_case=False, use_fast=False)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    model = model.get_encoder()

    D = int(getattr(model.config, "d_model", 0) or getattr(model.config, "hidden_size", 0))
    if D <= 0:
        raise RuntimeError("cannot infer embedding dim from model config")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dtype = np.float16 if (args.fp16 and device == "cuda") else np.float32
    mm = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(len(ids), D))

    items = []
    for nid in ids:
        s = seqs.get(nid)
        if s is None:
            raise RuntimeError(f"missing sequence: {nid}")
        items.append((nid, s, len(s)))
    items.sort(key=lambda x: x[2])

    buckets = {}
    for nid, s, L in items:
        k = (L // args.bucket) * args.bucket
        buckets.setdefault(k, []).append((nid, s))

    idx_map = {nid: i for i, nid in enumerate(ids)}
    pbar = tqdm(total=len(ids), desc="Embedding (ProtT5)")

    for k in sorted(buckets.keys()):
        group = buckets[k]
        for nid, s in group:
            parts = slide_seq(s, args.max_len, args.stride)
            v = embed_windows(model, tok, parts, device, args.max_len, args.win_bs, use_amp)
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

    if args.l2norm:
        mm2 = np.lib.format.open_memmap(out_path, mode="r+", dtype=dtype, shape=(len(ids), D))
        bs = 4096
        for i in range(0, len(ids), bs):
            x = np.array(mm2[i:i+bs], dtype=np.float32, copy=True)
            x = l2norm_rows(x)
            if dtype == np.float16:
                mm2[i:i+bs] = x.astype(np.float16, copy=False)
            else:
                mm2[i:i+bs] = x
        mm2.flush()

if __name__ == "__main__":
    main()