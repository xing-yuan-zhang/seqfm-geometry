import os
import argparse
import numpy as np

def read_edges3(path):
    es = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip().split("\t")
            if len(s) < 3:
                continue
            u, v, w = s[0], s[1], float(s[2])
            if u == v:
                continue
            es.append((u, v, w))
    return es

def write_edges3(path, es):
    with open(path, "w") as f:
        for u, v, w in es:
            f.write(u + "\t" + v + "\t" + f"{w:.6g}" + "\n")

def minmax01(xs):
    xs = np.asarray(xs, dtype=np.float64)
    lo = float(xs.min())
    hi = float(xs.max())
    if hi <= lo + 1e-12:
        return np.ones_like(xs) * 0.0
    return (xs - lo) / (hi - lo)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--string_edges", required=True)
    ap.add_argument("--fmppi_edges", required=True)
    ap.add_argument("--outdir", default="graphs")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--f", default="exp")
    ap.add_argument("--string_norm", default="minmax")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    S = read_edges3(args.string_edges)
    F = read_edges3(args.fmppi_edges)

    if args.string_norm == "minmax":
        sw = minmax01([w for _, _, w in S]).astype(np.float64)
        S = [(u, v, float(sw[i])) for i, (u, v, _) in enumerate(S)]
    elif args.string_norm == "clip01":
        S = [(u, v, float(max(0.0, min(1.0, w)))) for u, v, w in S]

    fm = {}
    for u, v, p in F:
        a, b = (u, v) if u < v else (v, u)
        fm[(a, b)] = float(p)

    out = []
    for u, v, w in S:
        a, b = (u, v) if u < v else (v, u)
        p = fm.get((a, b), 0.0)
        if args.f == "exp":
            g = np.exp(args.beta * p)
        else:
            g = 1.0 + args.beta * p
        out.append((u, v, float(w * g)))

    out_path = os.path.join(args.outdir, f"string_reweight_beta_{args.fmppi_edges.split('/')[-1].split('.')[0]}_{args.beta:g}_{args.f}.tsv")
    write_edges3(out_path, out)

if __name__ == "__main__":
    main()
