"""
Perform PR diffusion on the input graphs and evaluate results.

Output:
- {out_prefix}.top{topk}.txt: the top-k nodes ranked by diffusion scores.
- {out_prefix}.summary.json: summary of the evaluation results for this graph.

Call:
    from diffusion import run_diffusion
    run_diffusion(
        graphs_dir="graphs",
        seeds="seeds.txt",
        labels="labels.txt",
        out_dir="results",
    )
"""

from runner import run_folder

def run_diffusion(
    graphs_dir,
    seeds,
    labels,
    out_dir,
    pattern="*.tsv",
    nodes="",
    alpha=0.85,
    min_w=0.0,
    topk=200,
    max_iter=500,
    tol=1e-10,
):
    run_folder(
        graphs_dir=graphs_dir,
        pattern=pattern,
        seeds=seeds,
        labels=labels,
        out_dir=out_dir,
        nodes=nodes,
        alpha=alpha,
        min_w=min_w,
        topk=topk,
        max_iter=max_iter,
        tol=tol,
    )