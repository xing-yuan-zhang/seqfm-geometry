from pathlib import Path
import pandas as pd

def read_fa(p):
    m, k, buf = {}, None, []
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if k and buf:
                    m[k] = "".join(buf)
                k = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if k and buf:
            m[k] = "".join(buf)
    return m

def num_over_threshold(thres: int, fasta:dict[str,str]):
    count = 0
    for seq in fasta.values():
        if len(seq) > thres:
            count += 1
    return count

base = Path(__file__).resolve().parent
m = read_fa(base / "seqs.fasta")
c = num_over_threshold(2500, m)
print(f"{c}/{len(m)}")