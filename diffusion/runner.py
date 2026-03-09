import os
import glob
import json

from eval import eval_one_graph

def sort_rows(rows):
    rows.sort(key=lambda x: (-1e9 if x[4] is None else -x[4], -1e9 if x[5] is None else -x[5]))
    return rows

def run_folder(
    graphs_dir,
    pattern,
    seeds,
    labels,
    out_dir,
    nodes="",
    alpha=0.85,
    min_w=0.0,
    topk=200,
    max_iter=500,
    tol=1e-10,
):
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(graphs_dir, pattern)))
    if not paths:
        raise RuntimeError("no graphs matched")

    summaries = []
    for p in paths:
        base = os.path.basename(p)
        name = os.path.splitext(base)[0]
        out_prefix = os.path.join(out_dir, name)

        res = eval_one_graph(
            edges_path=p,
            seeds_path=seeds,
            labels_path=labels,
            out_prefix=out_prefix,
            nodes_whitelist_path=nodes,
            alpha=alpha,
            min_w=min_w,
            topk=topk,
            max_iter=max_iter,
            tol=tol,
        )
        summaries.append(res)

    with open(os.path.join(out_dir, "ALL.summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    rows = []
    for r in summaries:
        rows.append([
            os.path.basename(r["edges"]),
            r["n_nodes"],
            r["n_seeds_in_graph"],
            r["n_pos_eval"],
            r["auprc"],
            r["recall_at_k"],
            r["precision_at_k"],
            r["enrichment_factor_at_k"],
        ])

    rows = sort_rows(rows)

    with open(os.path.join(out_dir, "ALL.summary.tsv"), "w") as f:
        f.write("graph\tn_nodes\tn_seeds\tn_pos_eval\tauprc\trecall@k\tprecision@k\tenrich_factor@k\n")
        for row in rows:
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")

    return summaries