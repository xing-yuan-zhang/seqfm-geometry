import os
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F

def read_fasta(path):
    seq = {}
    k = None
    buf = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if k is not None:
                    seq[k] = "".join(buf)
                k = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
    if k is not None:
        seq[k] = "".join(buf)
    return seq

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

def append_edges(path, edges):
    with open(path, "a") as f:
        for u, v, w in edges:
            f.write(u + "\t" + v + "\t" + f"{w:.6g}" + "\n")

def get_env_int(name, default):
    v = os.environ.get(name, None)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default

def get_shard_range(N):
    shard_id = os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("SHARD_ID", None))
    num_shards = os.environ.get("SLURM_ARRAY_TASK_COUNT", os.environ.get("NUM_SHARDS", None))
    if shard_id is None or num_shards is None:
        return 0, N, -1, 1
    shard_id = int(shard_id)
    num_shards = max(1, int(num_shards))
    chunk = (N + num_shards - 1) // num_shards
    start = min(N, shard_id * chunk)
    end = min(N, (shard_id + 1) * chunk)
    return start, end, shard_id, num_shards

def get_amp_dtype(device):
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

class NullCtx:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc, tb):
        return False

def amp_ctx(device, dtype):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return NullCtx()

@torch.inference_mode()
def attn_cross_score(model, tok, seqA, seqB, device, linker_len=10, amp_dtype=None):
    linker = "X" * linker_len
    s = seqA + linker + seqB
    x = tok(s, return_tensors="pt", add_special_tokens=True)
    x = {k: v.to(device) for k, v in x.items()}
    with amp_ctx(device, amp_dtype):
        out = model(**x, output_attentions=True)

    if out.attentions is None or len(out.attentions) == 0:
        raise RuntimeError("No attentions returned. Use attn_implementation='eager'.")

    attn = out.attentions[-1][0].float().mean(dim=0)

    L = attn.shape[0]
    nA = len(seqA)
    nL = linker_len
    i0 = 1
    a0, a1 = i0, i0 + nA
    l0, l1 = a1, a1 + nL
    b0, b1 = l1, l1 + len(seqB)
    if b1 > L - 1:
        b1 = L - 1
    if a1 <= a0 or b1 <= b0:
        return 0.0

    ab = attn[a0:a1, b0:b1].mean().item()
    ba = attn[b0:b1, a0:a1].mean().item()
    s = 0.5 * (ab + ba)
    if not np.isfinite(s):
        s = 0.0
    return float(max(0.0, min(1.0, s)))

@torch.inference_mode()
def pll_score_batched(model, tok, seq, device, sample_k=0, seed=0, batch_size=16, amp_dtype=None):
    x = tok(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = x["input_ids"][0].to(device)
    attn = x["attention_mask"][0].to(device)
    L = int(attn.sum().item())
    ids = input_ids[:L].clone()
    attn = attn[:L]
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask_token_id")

    pos = list(range(1, L - 1))
    if len(pos) == 0:
        return 0.0

    if sample_k and sample_k < len(pos):
        rng = np.random.RandomState(seed)
        pos = rng.choice(np.array(pos), size=sample_k, replace=False).tolist()

    total = 0.0
    n = len(pos)

    for st in range(0, n, batch_size):
        chunk = pos[st:st + batch_size]
        B = len(chunk)
        batch_ids = ids.unsqueeze(0).repeat(B, 1)
        batch_attn = attn.unsqueeze(0).repeat(B, 1)
        row_idx = torch.arange(B, device=device)
        col_idx = torch.tensor(chunk, device=device, dtype=torch.long)
        true_ids = batch_ids[row_idx, col_idx].clone()
        batch_ids[row_idx, col_idx] = mask_id

        with amp_ctx(device, amp_dtype):
            out = model(input_ids=batch_ids, attention_mask=batch_attn)
            logits = out.logits[row_idx, col_idx].float()

        lp = F.log_softmax(logits, dim=-1)
        total += lp[row_idx, true_ids].sum().item()

    return float(total / max(1, n))

@torch.inference_mode()
def pair_pll_diff(model, tok, seqA, seqB, device, linker_len=10, sample_k=0, seed=0,
                  batch_size=16, amp_dtype=None, single_pll_cache=None):
    linker = "X" * linker_len
    sAB = seqA + linker + seqB

    pll_ab = pll_score_batched(
        model, tok, sAB, device,
        sample_k=sample_k,
        seed=seed,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )

    keyA = (seqA, sample_k, seed + 1)
    keyB = (seqB, sample_k, seed + 2)

    if single_pll_cache is not None and keyA in single_pll_cache:
        pll_a = single_pll_cache[keyA]
    else:
        pll_a = pll_score_batched(
            model, tok, seqA, device,
            sample_k=sample_k,
            seed=seed + 1,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
        if single_pll_cache is not None:
            single_pll_cache[keyA] = pll_a

    if single_pll_cache is not None and keyB in single_pll_cache:
        pll_b = single_pll_cache[keyB]
    else:
        pll_b = pll_score_batched(
            model, tok, seqB, device,
            sample_k=sample_k,
            seed=seed + 2,
            batch_size=batch_size,
            amp_dtype=amp_dtype,
        )
        if single_pll_cache is not None:
            single_pll_cache[keyB] = pll_b

    return float(pll_ab - pll_a - pll_b)

@torch.inference_mode()
def logprob_true_tokens_batch(logits, input_ids, pos_idx):
    if len(pos_idx) == 0:
        return torch.empty(logits.shape[0], device=logits.device)
    pos = torch.tensor(pos_idx, device=logits.device, dtype=torch.long)
    x = logits.index_select(1, pos).float()
    lp = F.log_softmax(x, dim=-1)
    true = input_ids[pos].unsqueeze(0).unsqueeze(-1).expand(lp.shape[0], -1, -1)
    return lp.gather(-1, true).squeeze(-1).mean(dim=1)

@torch.inference_mode()
def perturb_coupling(model, tok, seqA, seqB, device, linker_len=10,
                     perturb_k=8, score_k=64, seed=0, amp_dtype=None):
    linker = "X" * linker_len
    s = seqA + linker + seqB
    x = tok(s, return_tensors="pt", add_special_tokens=True)
    input_ids = x["input_ids"][0].to(device)
    attn = x["attention_mask"][0].to(device)
    L = int(attn.sum().item())
    input_ids = input_ids[:L]
    attn = attn[:L]

    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask_token_id")

    nA = len(seqA)
    nL = linker_len
    i0 = 1
    a0, a1 = i0, i0 + nA
    l0, l1 = a1, a1 + nL
    b0, b1 = l1, l1 + len(seqB)
    if b1 > L - 1:
        b1 = L - 1
    a1 = min(a1, L - 1)
    if a1 <= a0 or b1 <= b0:
        return 0.0

    rng = np.random.RandomState(seed)

    Apos = list(range(a0, a1))
    Bpos = list(range(b0, b1))

    if score_k and score_k < len(Bpos):
        Bscore = rng.choice(np.array(Bpos), size=score_k, replace=False).tolist()
    else:
        Bscore = Bpos

    if score_k and score_k < len(Apos):
        Ascore = rng.choice(np.array(Apos), size=score_k, replace=False).tolist()
    else:
        Ascore = Apos

    with amp_ctx(device, amp_dtype):
        out0 = model(input_ids=input_ids.unsqueeze(0), attention_mask=attn.unsqueeze(0))
    logits0 = out0.logits
    base_B = float(logprob_true_tokens_batch(logits0, input_ids, Bscore)[0].item())
    base_A = float(logprob_true_tokens_batch(logits0, input_ids, Ascore)[0].item())

    if perturb_k and perturb_k < len(Apos):
        Amask = rng.choice(np.array(Apos), size=perturb_k, replace=False).tolist()
    else:
        Amask = Apos

    if perturb_k and perturb_k < len(Bpos):
        Bmask = rng.choice(np.array(Bpos), size=perturb_k, replace=False).tolist()
    else:
        Bmask = Bpos

    drop_AB = []
    if len(Amask) > 0:
        B = len(Amask)
        batch_ids = input_ids.unsqueeze(0).repeat(B, 1)
        batch_attn = attn.unsqueeze(0).repeat(B, 1)
        row_idx = torch.arange(B, device=device)
        col_idx = torch.tensor(Amask, device=device, dtype=torch.long)
        batch_ids[row_idx, col_idx] = mask_id
        with amp_ctx(device, amp_dtype):
            out = model(input_ids=batch_ids, attention_mask=batch_attn)
        lpB = logprob_true_tokens_batch(out.logits, input_ids, Bscore).cpu().numpy()
        drop_AB = np.maximum(0.0, base_B - lpB).tolist()

    drop_BA = []
    if len(Bmask) > 0:
        B = len(Bmask)
        batch_ids = input_ids.unsqueeze(0).repeat(B, 1)
        batch_attn = attn.unsqueeze(0).repeat(B, 1)
        row_idx = torch.arange(B, device=device)
        col_idx = torch.tensor(Bmask, device=device, dtype=torch.long)
        batch_ids[row_idx, col_idx] = mask_id
        with amp_ctx(device, amp_dtype):
            out = model(input_ids=batch_ids, attention_mask=batch_attn)
        lpA = logprob_true_tokens_batch(out.logits, input_ids, Ascore).cpu().numpy()
        drop_BA = np.maximum(0.0, base_A - lpA).tolist()

    s1 = float(np.mean(drop_AB)) if drop_AB else 0.0
    s2 = float(np.mean(drop_BA)) if drop_BA else 0.0
    return 0.5 * (s1 + s2)

def norm_to_prob(scores, mode, temp=1.0):
    scores = np.asarray(scores, dtype=np.float64)
    if mode == "attn_clip01":
        return np.clip(scores, 0.0, 1.0)
    if mode == "minmax":
        lo, hi = float(scores.min()), float(scores.max())
        if hi <= lo + 1e-12:
            return np.zeros_like(scores)
        return (scores - lo) / (hi - lo)
    mu = float(scores.mean())
    sd = float(scores.std() + 1e-12)
    z = (scores - mu) / sd
    return 1.0 / (1.0 + np.exp(-z / max(temp, 1e-12)))

def load_model_and_tokenizer(args, device):
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

    tok = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=False)

    amp_dtype = get_amp_dtype(device)

    if args.method in ["pll", "perturb"]:
        model_kwargs = {}
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = amp_dtype
            model_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForMaskedLM.from_pretrained(args.model_name, **model_kwargs).to(device).eval()
    else:
        model_kwargs = {"attn_implementation": "eager"}
        if device.type == "cuda":
            model_kwargs["torch_dtype"] = amp_dtype
        model = AutoModel.from_pretrained(args.model_name, **model_kwargs).to(device).eval()

    return tok, model, amp_dtype

def maybe_enable_cuda_fastpath():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def dedup_edges_max(raw_edges):
    dct = {}
    for u, v, w in raw_edges:
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in dct or w > dct[key]:
            dct[key] = w
    out = [(u, v, w) for (u, v), w in dct.items() if w > 0]
    out.sort()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--outdir", default="graphs")
    ap.add_argument("--method", choices=["attn", "pll", "perturb"], default="attn")
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cand_M", type=int, default=500)
    ap.add_argument("--topm", type=int, default=20)
    ap.add_argument("--linker_len", type=int, default=10)

    ap.add_argument("--pll_sample_k", type=int, default=0)
    ap.add_argument("--pll_seed", type=int, default=0)
    ap.add_argument("--pll_norm", choices=["sigmoid_z", "minmax"], default="sigmoid_z")

    ap.add_argument("--perturb_k", type=int, default=8)
    ap.add_argument("--score_k", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--sigmoid_temp", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    maybe_enable_cuda_fastpath()

    z = np.load(args.emb_npz, allow_pickle=True)
    ids = list(z["ids"])
    emb = np.array(z["emb"], dtype=np.float32)
    id2i = {k: i for i, k in enumerate(ids)}

    nodes = read_ids(args.nodes)
    nodes = [x for x in nodes if x in id2i]

    seq = read_fasta(args.fasta)
    nodes = [x for x in nodes if x in seq and len(seq[x]) > 0]

    X = l2norm(emb[[id2i[x] for x in nodes]]).astype(np.float32)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    tok, model, amp_dtype = load_model_and_tokenizer(args, device)

    G = X @ X.T
    np.fill_diagonal(G, -1.0)

    N = len(nodes)
    M = min(args.cand_M, N - 1)
    topm = min(args.topm, M)

    start, end, shard_id, num_shards = get_shard_range(N)

    if args.method == "pll":
        pll_batch_size = get_env_int("PLL_BATCH_SIZE", 16)
    else:
        pll_batch_size = get_env_int("PLL_BATCH_SIZE", 16)

    flush_every = get_env_int("FLUSH_EVERY", 128)

    base_name = f"fmppi_unsup_{args.method}_M{M}_m{topm}"
    if shard_id >= 0:
        out_path = os.path.join(args.outdir, f"{base_name}.part{shard_id:04d}.tsv")
        meta_path = os.path.join(args.outdir, f"{base_name}.part{shard_id:04d}.meta.txt")
    else:
        out_path = os.path.join(args.outdir, f"{base_name}.tsv")
        meta_path = os.path.join(args.outdir, f"{base_name}.meta.txt")

    if os.path.exists(out_path):
        os.remove(out_path)

    raw_edges_buf = []
    single_pll_cache = {} if args.method == "pll" else None

    for i in range(start, end):
        cand_idx = np.argpartition(-G[i], M - 1)[:M]
        u = nodes[i]
        sA = seq[u]

        scored = []
        for j in cand_idx:
            v = nodes[int(j)]
            sB = seq[v]

            if args.method == "attn":
                w = attn_cross_score(
                    model, tok, sA, sB, device,
                    linker_len=args.linker_len,
                    amp_dtype=amp_dtype,
                )
            elif args.method == "pll":
                w = pair_pll_diff(
                    model, tok, sA, sB, device,
                    linker_len=args.linker_len,
                    sample_k=args.pll_sample_k,
                    seed=args.pll_seed + i,
                    batch_size=pll_batch_size,
                    amp_dtype=amp_dtype,
                    single_pll_cache=single_pll_cache,
                )
            else:
                w = perturb_coupling(
                    model, tok, sA, sB, device,
                    linker_len=args.linker_len,
                    perturb_k=args.perturb_k,
                    score_k=args.score_k,
                    seed=(args.seed + 1000003 * i + 97 * int(j)) % (2**32 - 1),
                    amp_dtype=amp_dtype,
                )

            scored.append((v, float(w)))

        scores = np.array([w for _, w in scored], dtype=np.float64)

        if args.method == "attn":
            probs = norm_to_prob(scores, "attn_clip01", temp=args.sigmoid_temp)
        elif args.method == "pll":
            probs = norm_to_prob(
                scores,
                "minmax" if args.pll_norm == "minmax" else "sigmoid_z",
                temp=args.sigmoid_temp,
            )
        else:
            probs = norm_to_prob(scores, "sigmoid_z", temp=args.sigmoid_temp)

        sel = np.argpartition(-probs, topm - 1)[:topm]
        for k in sel:
            v = scored[int(k)][0]
            p = float(probs[int(k)])
            raw_edges_buf.append((u, v, p))

        local_i = i - start + 1
        if local_i % flush_every == 0 or i == end - 1:
            out_chunk = dedup_edges_max(raw_edges_buf)
            append_edges(out_path, out_chunk)
            raw_edges_buf = []

            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"[{local_i}/{max(1, end - start)}] shard={shard_id} node={i} flushed={len(out_chunk)}", flush=True)

    with open(meta_path, "w") as f:
        f.write(f"emb_npz\t{args.emb_npz}\n")
        f.write(f"fasta\t{args.fasta}\n")
        f.write(f"nodes\t{args.nodes}\n")
        f.write(f"model_name\t{args.model_name}\n")
        f.write(f"method\t{args.method}\n")
        f.write(f"cand_M\t{M}\n")
        f.write(f"topm\t{topm}\n")
        f.write(f"linker_len\t{args.linker_len}\n")
        f.write(f"sigmoid_temp\t{args.sigmoid_temp}\n")
        f.write(f"pll_sample_k\t{args.pll_sample_k}\n")
        f.write(f"pll_norm\t{args.pll_norm}\n")
        f.write(f"perturb_k\t{args.perturb_k}\n")
        f.write(f"score_k\t{args.score_k}\n")
        f.write(f"seed\t{args.seed}\n")
        f.write(f"pll_seed\t{args.pll_seed}\n")
        f.write(f"device\t{device}\n")
        f.write(f"amp_dtype\t{amp_dtype}\n")
        f.write(f"start\t{start}\n")
        f.write(f"end\t{end}\n")
        f.write(f"shard_id\t{shard_id}\n")
        f.write(f"num_shards\t{num_shards}\n")
        f.write(f"pll_batch_size\t{pll_batch_size}\n")
        f.write(f"flush_every\t{flush_every}\n")

if __name__ == "__main__":
    main()