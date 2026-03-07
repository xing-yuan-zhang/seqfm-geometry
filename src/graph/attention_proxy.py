import os
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

@torch.no_grad()
def attn_cross_score(model, tok, seqA, seqB, device, linker_len=10):
    linker = "X" * linker_len
    s = seqA + linker + seqB
    x = tok(s, return_tensors="pt", add_special_tokens=True)
    x = {k: v.to(device) for k, v in x.items()}
    out = model(**x, output_attentions=True)

    if out.attentions is None or len(out.attentions) == 0:
        raise RuntimeError("No attentions returned. Use attn_implementation='eager'.")

    attn = out.attentions[-1][0].mean(dim=0)

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

@torch.no_grad()
def pll_score(model, tok, seq, device, sample_k=0, seed=0):
    x = tok(seq, return_tensors="pt", add_special_tokens=True)
    input_ids = x["input_ids"][0].to(device)
    attn = x["attention_mask"][0].to(device)
    L = int(attn.sum().item())
    ids = input_ids[:L].clone()
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask_token_id")

    pos = list(range(1, L-1))
    if sample_k and sample_k < len(pos):
        rng = np.random.RandomState(seed)
        pos = rng.choice(np.array(pos), size=sample_k, replace=False).tolist()

    total = 0.0
    for i in pos:
        true_id = int(ids[i].item())
        masked = ids.clone()
        masked[i] = mask_id
        out = model(input_ids=masked.unsqueeze(0), attention_mask=attn[:L].unsqueeze(0))
        logits = out.logits[0, i]
        lp = F.log_softmax(logits, dim=-1)[true_id].item()
        total += lp
    return float(total / max(1, len(pos)))

@torch.no_grad()
def pair_pll_diff(model, tok, seqA, seqB, device, linker_len=10, sample_k=0, seed=0):
    linker = "X" * linker_len
    sAB = seqA + linker + seqB
    pll_ab = pll_score(model, tok, sAB, device, sample_k=sample_k, seed=seed)
    pll_a = pll_score(model, tok, seqA, device, sample_k=sample_k, seed=seed+1)
    pll_b = pll_score(model, tok, seqB, device, sample_k=sample_k, seed=seed+2)
    return float(pll_ab - pll_a - pll_b)

@torch.no_grad()
def logprob_true_tokens(logits, input_ids, pos_idx):
    if len(pos_idx) == 0:
        return 0.0
    lp = F.log_softmax(logits[pos_idx], dim=-1)
    true = input_ids[pos_idx].unsqueeze(1)
    return float(lp.gather(1, true).mean().item())

@torch.no_grad()
def perturb_coupling(model, tok, seqA, seqB, device, linker_len=10,
                     perturb_k=8, score_k=64, seed=0):
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

    out0 = model(input_ids=input_ids.unsqueeze(0), attention_mask=attn.unsqueeze(0))
    logits0 = out0.logits[0]

    base_B = logprob_true_tokens(logits0, input_ids, Bscore)
    base_A = logprob_true_tokens(logits0, input_ids, Ascore)

    if perturb_k and perturb_k < len(Apos):
        Amask = rng.choice(np.array(Apos), size=perturb_k, replace=False).tolist()
    else:
        Amask = Apos

    if perturb_k and perturb_k < len(Bpos):
        Bmask = rng.choice(np.array(Bpos), size=perturb_k, replace=False).tolist()
    else:
        Bmask = Bpos

    drop_AB = []
    for i in Amask:
        ids = input_ids.clone()
        ids[i] = mask_id
        out = model(input_ids=ids.unsqueeze(0), attention_mask=attn.unsqueeze(0))
        logits = out.logits[0]
        lpB = logprob_true_tokens(logits, input_ids, Bscore)
        drop_AB.append(max(0.0, base_B - lpB))

    drop_BA = []
    for j in Bmask:
        ids = input_ids.clone()
        ids[j] = mask_id
        out = model(input_ids=ids.unsqueeze(0), attention_mask=attn.unsqueeze(0))
        logits = out.logits[0]
        lpA = logprob_true_tokens(logits, input_ids, Ascore)
        drop_BA.append(max(0.0, base_A - lpA))

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

    z = np.load(args.emb_npz, allow_pickle=True)
    ids = list(z["ids"])
    emb = np.array(z["emb"], dtype=np.float32)
    id2i = {k:i for i,k in enumerate(ids)}

    nodes = read_ids(args.nodes)
    nodes = [x for x in nodes if x in id2i]

    seq = read_fasta(args.fasta)
    nodes = [x for x in nodes if x in seq and len(seq[x]) > 0]

    X = l2norm(emb[[id2i[x] for x in nodes]]).astype(np.float32)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

    tok = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    if args.method in ["pll", "perturb"]:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name,
            attn_implementation="eager",
        ).to(device).eval()
    else:
        model = AutoModel.from_pretrained(
            args.model_name,
            attn_implementation="eager",
        ).to(device).eval()

    G = X @ X.T
    np.fill_diagonal(G, -1.0)

    N = len(nodes)
    M = min(args.cand_M, N - 1)
    topm = min(args.topm, M)

    raw_edges = []
    for i in range(N):
        cand_idx = np.argpartition(-G[i], M)[:M]
        u = nodes[i]
        sA = seq[u]

        scored = []
        for j in cand_idx:
            v = nodes[int(j)]
            sB = seq[v]
            if args.method == "attn":
                w = attn_cross_score(model, tok, sA, sB, device, linker_len=args.linker_len)
            elif args.method == "pll":
                w = pair_pll_diff(model, tok, sA, sB, device, linker_len=args.linker_len,
                                  sample_k=args.pll_sample_k, seed=args.pll_seed + i)
            else:
                w = perturb_coupling(model, tok, sA, sB, device,
                                     linker_len=args.linker_len,
                                     perturb_k=args.perturb_k,
                                     score_k=args.score_k,
                                     seed=args.seed + 1000003 * i + 97 * int(j))
            scored.append((v, float(w)))

        scores = np.array([w for _, w in scored], dtype=np.float64)

        if args.method == "attn":
            probs = norm_to_prob(scores, "attn_clip01", temp=args.sigmoid_temp)
        elif args.method == "pll":
            probs = norm_to_prob(scores, "minmax" if args.pll_norm == "minmax" else "sigmoid_z",
                                 temp=args.sigmoid_temp)
        else:
            probs = norm_to_prob(scores, "sigmoid_z", temp=args.sigmoid_temp)

        sel = np.argpartition(-probs, topm)[:topm]
        for k in sel:
            v = scored[int(k)][0]
            p = float(probs[int(k)])
            raw_edges.append((u, v, p))

    dct = {}
    for u, v, w in raw_edges:
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in dct or w > dct[key]:
            dct[key] = w

    out = [(u, v, w) for (u, v), w in dct.items() if w > 0]
    out.sort()

    out_path = os.path.join(args.outdir, f"fmppi_unsup_{args.method}_M{M}_m{topm}.tsv")
    write_edges(out_path, out)

    meta_path = os.path.join(args.outdir, f"fmppi_unsup_{args.method}_M{M}_m{topm}.meta.txt")
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

if __name__ == "__main__":
    main()