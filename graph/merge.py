import os
import glob
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if len(files) == 0:
        raise RuntimeError()

    d = {}
    n_in = 0
    for fp in files:
        with open(fp, "r") as f:
            for line in f:
                s = line.rstrip("\n").split("\t")
                if len(s) < 3:
                    continue
                u, v = s[0], s[1]
                if u == v:
                    continue
                try:
                    w = float(s[2])
                except Exception:
                    continue
                a, b = (u, v) if u < v else (v, u)
                key = (a, b)
                if key not in d or w > d[key]:
                    d[key] = w
                n_in += 1

    out = [(u, v, w) for (u, v), w in d.items() if w > 0]
    out.sort()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for u, v, w in out:
            f.write(u + "\t" + v + "\t" + f"{w:.6g}" + "\n")

if __name__ == "__main__":
    main()