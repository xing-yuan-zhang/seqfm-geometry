"""
Preprocessing sequences for ESM2/ProtT5 embeddings.

Call:
    from seqs_parser import seqs_parser
    seqs_parser(
        info="string_info.tsv",
        seqs="9606.protein.sequences.v12.0.fa",
        outdir="outputs"
    )
"""

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

def seqs_parser(info, seqs, outdir="."):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    info_df = pd.read_csv(info, sep="\t", dtype=str)
    sids = info_df["string_id"].astype(str).tolist()
    seqm = read_fa(seqs)

    keep = []
    for sid in sids:
        seq = seqm.get(sid)
        keep.append((sid, seq)) if seq else None

    with open(outdir / "node_ids.txt", "w", encoding="utf-8") as f:
        for sid, _ in keep:
            f.write(sid + "\n")

    with open(outdir / "seqs.fasta", "w", encoding="utf-8") as f:
        for sid, seq in keep:
            f.write(">" + sid + "\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

    with open(outdir / "seqs_length.tsv", "w", encoding="utf-8") as f:
        for seq in keep:
            f.write("\t" + str(len(seq)) + "\n")

    return {
        "outdir": outdir,
        "n_input_ids": len(sids),
        "n_kept": len(keep),
        "keep": keep,
    }