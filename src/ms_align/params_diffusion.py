import argparse
import time
import pickle
from collections import deque
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import requests

API = "https://rest.uniprot.org"


def clean_node_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.split("-").str[0]
        .str.split(".").str[0]
    )


def submit_mapping_job(acc_list):
    r = requests.post(
        f"{API}/idmapping/run",
        data={
            "from": "UniProtKB_AC-ID",
            "to": "UniProtKB",
            "ids": ",".join(acc_list),
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["jobId"]


def wait_for_job(job_id, sleep_s=1.5):
    while True:
        r = requests.get(f"{API}/idmapping/status/{job_id}", timeout=60)
        r.raise_for_status()
        js = r.json()
        if js.get("jobStatus") in ("NEW", "RUNNING"):
            time.sleep(sleep_s)
            continue
        return js


def fetch_all_results_tsv(job_id, fields="accession,gene_primary,protein_name"):
    url = f"{API}/idmapping/uniprotkb/results/{job_id}"
    params = {"format": "tsv", "fields": fields}
    buf = []
    first = True

    while url:
        r = requests.get(url, params=params if first else None, timeout=120)
        r.raise_for_status()
        txt = r.text
        if first:
            buf.append(txt)
            first = False
        else:
            lines = txt.splitlines()
            if len(lines) > 1:
                buf.append("\n".join(lines[1:]) + "\n")
        url = r.links.get("next", {}).get("url")

    return "".join(buf)


def parse_mapping_tsv(tsv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(tsv_text), sep="\t")
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("entry", "accession"):
            ren[c] = "accession"
        elif "gene" in lc and "primary" in lc:
            ren[c] = "gene_symbol"
        elif "protein" in lc and "name" in lc:
            ren[c] = "protein_name"
    df = df.rename(columns=ren)

    keep = [c for c in ["accession", "gene_symbol", "protein_name"] if c in df.columns]
    return df[keep].drop_duplicates("accession")


def fetch_uniprot_names(accessions, batch_size=500) -> pd.DataFrame:
    acc = clean_node_series(pd.Series(accessions))
    uniq = []
    seen = set()
    for x in acc.tolist():
        if not x or str(x).lower() == "nan":
            continue
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    rows = []
    for i in range(0, len(uniq), batch_size):
        batch = uniq[i:i + batch_size]
        job = submit_mapping_job(batch)
        wait_for_job(job)
        tsv = fetch_all_results_tsv(job)
        rows.append(parse_mapping_tsv(tsv))

    if not rows:
        return pd.DataFrame(columns=["accession", "gene_symbol", "protein_name"])

    return pd.concat(rows, ignore_index=True).drop_duplicates("accession")


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_graph(edges_tsv: Path, pkl_graph: Path | None):
    if pkl_graph and pkl_graph.exists():
        g = pickle.load(open(pkl_graph, "rb"))
        if not isinstance(g, nx.Graph):
            raise ValueError("graph pkl is not networkx Graph")
        return g

    e = pd.read_csv(edges_tsv, sep="\t", dtype=str)
    if "weight" not in e.columns:
        e["weight"] = 1.0
    e["weight"] = pd.to_numeric(e["weight"], errors="coerce").fillna(1.0)

    g = nx.Graph()
    for r in e.itertuples(index=False):
        a = getattr(r, "entry_a")
        b = getattr(r, "entry_b")
        if a == b:
            continue
        g.add_edge(str(a), str(b), weight=float(getattr(r, "weight")))
    return g


def multi_source_hops(g: nx.Graph, seeds: list[str], cutoff: int | None):
    hops = {}
    q = deque()

    for s in seeds:
        if s in g:
            hops[s] = 0
            q.append(s)

    while q:
        u = q.popleft()
        if cutoff is not None and hops[u] >= cutoff:
            continue
        for v in g.neighbors(u):
            if v not in hops:
                hops[v] = hops[u] + 1
                q.append(v)

    return hops


def build_rank_map(diff_df: pd.DataFrame):
    d = diff_df[["node", "score"]].copy()
    d["node"] = d["node"].astype(str).str.strip()
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.dropna(subset=["score"]).sort_values("score", ascending=False)
    d["rank"] = range(1, len(d) + 1)
    d["key"] = d["node"].str.split("-").str[0]
    return dict(zip(d["key"], d["rank"]))


def node_key(x: str) -> str:
    return str(x).strip().split("-")[0]


def build_display_maps(mapping_df: pd.DataFrame):
    m = mapping_df.copy()
    if "accession" not in m.columns:
        return {}, {}
    m["accession"] = clean_node_series(m["accession"])
    gene = dict(zip(m["accession"], m.get("gene_symbol", pd.Series(index=m.index, dtype=str)).astype(str)))
    prot = dict(zip(m["accession"], m.get("protein_name", pd.Series(index=m.index, dtype=str)).astype(str)))

    overrides = {
        "C1orf39": "TOCA-1",
        "TRIP10": "CIP4",
        "FNBP1": "FBP17",
        "FNBP1L": "TOCA-1",
    }

    def _fix(sym: str):
        s = str(sym) if sym is not None else ""
        s = s.strip()
        return overrides.get(s, s) if s else s

    gene = {k: _fix(v) for k, v in gene.items() if k and str(k).lower() != "nan"}
    prot = {k: v for k, v in prot.items() if k and str(k).lower() != "nan"}
    return gene, prot


def get_display_name(n: str, gene_map: dict, fallback=True):
    k = node_key(n)
    gs = gene_map.get(k)
    if gs and str(gs).lower() != "nan":
        return str(gs)
    return str(n) if fallback else str(k)


def pick_llps_col(nodes_df: pd.DataFrame):
    cols = list(nodes_df.columns)
    cand = []
    for c in cols:
        lc = str(c).lower()
        if "llps" in lc and ("any" in lc or lc.endswith("llps") or lc == "llps_any"):
            cand.append(c)
    if "is_LLPS_any" in cols:
        return "is_LLPS_any"
    if "LLPS_any" in cols:
        return "LLPS_any"
    return cand[0] if cand else None


def viz_colors_by_rank(g, rank_map, is_seed_map):
    cmap = plt.cm.Blues
    out = []
    for n in g.nodes():
        if int(is_seed_map.get(n, 0)) == 1:
            out.append("red")
            continue
        r = rank_map.get(node_key(n))
        if r is None:
            out.append("lightgray")
        elif r <= 50:
            out.append(cmap(0.9))
        elif r <= 100:
            out.append(cmap(0.7))
        elif r <= 200:
            out.append(cmap(0.5))
        else:
            out.append(cmap(0.3))
    return out


def plot_hop_subgraph_structure_rank_shaded(g, nodes_df, hops, rank_map, outdir, max_hop=2, layout_seed=42):
    is_seed = dict(zip(nodes_df["entry"].astype(str), nodes_df["is_seed"].astype(int)))
    keep = [n for n, h in hops.items() if h <= max_hop]
    sg = g.subgraph(keep).copy()
    if sg.number_of_nodes() == 0:
        raise ValueError("empty hop subgraph")

    k = 0.35
    shrink = 0.75
    node_sz = 60
    seed_sz = 110
    if int(max_hop) == 1:
        k = 0.06
        shrink = 0.55
        node_sz = 180
        seed_sz = 420

    pos = nx.spring_layout(sg, seed=layout_seed, weight="weight", k=k)
    for n in pos:
        pos[n] = pos[n] * shrink

    colors = viz_colors_by_rank(sg, rank_map, is_seed)
    w = np.array([sg[u][v].get("weight", 1.0) for u, v in sg.edges()], float)

    if len(w):
        lo, hi = float(w.min()), float(w.max())
        widths = (0.5 + 2.0 * (w - lo) / (hi - lo)) if hi > lo else np.full_like(w, 1.0)
    else:
        widths = []

    seeds = [n for n in sg.nodes() if int(is_seed.get(n, 0)) == 1]
    others = [n for n in sg.nodes() if n not in set(seeds)]
    c_map = dict(zip(list(sg.nodes()), colors))

    plt.figure(figsize=(9, 9))
    nx.draw_networkx_edges(sg, pos, width=widths, alpha=0.25)
    if others:
        nx.draw_networkx_nodes(
            sg, pos,
            nodelist=others,
            node_color=[c_map[n] for n in others],
            node_size=node_sz,
            linewidths=0,
        )
    if seeds:
        nx.draw_networkx_nodes(
            sg, pos,
            nodelist=seeds,
            node_color=[c_map[n] for n in seeds],
            node_size=seed_sz,
            linewidths=0,
        )

    plt.title(f"Hop subgraph (<= {max_hop}) with rank shading")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "fig1_hop_subgraph_rank_shaded.png", dpi=300)
    plt.close()


def plot_rank_score_curve(diff_df, outdir):
    d = diff_df.sort_values("score", ascending=False).reset_index(drop=True)
    d["rank"] = np.arange(1, len(d) + 1)
    y = np.clip(pd.to_numeric(d["score"], errors="coerce").to_numpy(), 1e-18, None)

    plt.figure(figsize=(8, 5.5))
    plt.plot(d["rank"], y)
    plt.yscale("log")
    plt.xlabel("Rank")
    plt.ylabel("PPR score (log)")
    plt.tight_layout()
    plt.savefig(outdir / "fig2_rank_score_log.png", dpi=300)
    plt.close()


def plot_score_vs_hop_boxplot(diff_df, hops, outdir, max_hop_show=4):
    d = diff_df.copy()
    d["node"] = d["node"].astype(str)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d["hop"] = d["node"].map(hops)
    d = d.dropna(subset=["score", "hop"])
    d["hop"] = d["hop"].astype(int)
    d = d[d["hop"] <= max_hop_show]
    if d.empty:
        raise ValueError("no nodes for hop boxplot")

    hs = sorted(d["hop"].unique())
    groups = [np.clip(d.loc[d["hop"] == h, "score"].to_numpy(), 1e-18, None) for h in hs]

    plt.figure(figsize=(8, 5.5))
    plt.boxplot(groups, labels=[str(h) for h in hs], showfliers=False)
    plt.yscale("log")
    plt.xlabel("Hop distance")
    plt.ylabel("PPR score (log)")
    plt.tight_layout()
    plt.savefig(outdir / "fig3_score_vs_hop_boxplot.png", dpi=300)
    plt.close()


def plot_topk_induced_subgraph(g, nodes_df, diff_df, gene_map, outdir, topk=50, layout_seed=42):
    d = diff_df.copy()
    d["node"] = d["node"].astype(str)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.dropna(subset=["score"]).sort_values("score", ascending=False)

    top = []
    seen = set()
    for n in d["node"].tolist():
        k0 = node_key(n)
        if k0 in g and k0 not in seen:
            top.append(k0)
            seen.add(k0)
        if len(top) >= int(topk):
            break

    if not top:
        raise ValueError("empty topk induced subgraph")

    sg = g.subgraph(top).copy()

    is_seed = dict(zip(nodes_df["entry"].astype(str), nodes_df["is_seed"].astype(int)))
    seeds = [n for n in sg.nodes() if int(is_seed.get(n, 0)) == 1]
    others = [n for n in sg.nodes() if n not in set(seeds)]

    k = 0.05
    shrink = 0.6

    rng = np.random.default_rng(int(layout_seed))
    init_pos = {}

    if seeds:
        r0 = 0.02
        for i, s in enumerate(seeds):
            ang = 2 * np.pi * (i / max(1, len(seeds)))
            init_pos[s] = np.array([r0 * np.cos(ang), r0 * np.sin(ang)])

    for n in sg.nodes():
        if n in init_pos:
            continue
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.6, 1.0)
        init_pos[n] = np.array([rad * np.cos(ang), rad * np.sin(ang)])

    if seeds:
        pos = nx.spring_layout(sg, seed=layout_seed, weight="weight", k=k, pos=init_pos, fixed=seeds)
    else:
        pos = nx.spring_layout(sg, seed=layout_seed, weight="weight", k=k, pos=init_pos)

    for n in pos:
        pos[n] = pos[n] * shrink

    w = np.array([sg[u][v].get("weight", 1.0) for u, v in sg.edges()], float)
    if len(w):
        lo, hi = float(w.min()), float(w.max())
        widths = (0.6 + 2.5 * (w - lo) / (hi - lo)) if hi > lo else np.full_like(w, 1.2)
    else:
        widths = []

    labels = {n: get_display_name(n, gene_map, fallback=True) for n in sg.nodes()}

    plt.figure(figsize=(9, 9))
    nx.draw_networkx_edges(
        sg, pos,
        width=widths,
        edge_color="#3F3F3F",
        alpha=0.18,
    )
    if others:
        nx.draw_networkx_nodes(sg, pos, nodelist=others, node_color=plt.cm.Blues(0.7), node_size=800, linewidths=0)
    if seeds:
        nx.draw_networkx_nodes(sg, pos, nodelist=seeds, node_color="red", node_size=1200, linewidths=0)

    nx.draw_networkx_labels(sg, pos, labels=labels, font_size=14)
    plt.title(f"Top-{int(topk):02d} induced subgraph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / f"fig4_top{int(topk):02d}_induced.png", dpi=300)
    plt.close()


def plot_topk_barh(diff_df, nodes_df, gene_map, outdir, topk=25):
    d = diff_df.copy()
    d["node"] = d["node"].astype(str)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.dropna(subset=["score"]).sort_values("score", ascending=False)

    keep, seen = [], set()
    for n in d["node"].tolist():
        k0 = node_key(n)
        if k0 not in seen:
            keep.append(n)
            seen.add(k0)
        if len(keep) >= int(topk):
            break

    sub = d.set_index("node").loc[keep].reset_index()
    sub["node_key"] = sub["node"].map(node_key)
    sub["label"] = sub["node_key"].map(lambda x: get_display_name(x, gene_map, fallback=True))

    llps_col = pick_llps_col(nodes_df)
    llps = {}
    if llps_col:
        tmp = nodes_df[["entry", llps_col]].copy()
        tmp["entry"] = tmp["entry"].astype(str)
        tmp[llps_col] = pd.to_numeric(tmp[llps_col], errors="coerce").fillna(0).astype(int)
        llps = dict(zip(tmp["entry"], tmp[llps_col]))

    sub["is_llps_any"] = sub["node_key"].map(lambda x: int(llps.get(x, 0)) == 1)
    sub = sub.iloc[::-1].reset_index(drop=True)

    y = np.arange(len(sub))
    colors = ["#1b5e20" if b else "#a5d6a7" for b in sub["is_llps_any"]]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(
        y,
        sub["score"].to_numpy(),
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"], fontsize=13)
    ax.tick_params(axis="x", labelsize=12)
    ax.set_xlabel("PPR score", fontsize=14)

    for spine in ax.spines.values():
        spine.set_visible(False)

    mx = float(sub["score"].max())
    pad = mx * 0.015 if mx > 0 else 0.01
    for i, b in enumerate(bars):
        ax.text(
            b.get_width() + pad,
            b.get_y() + b.get_height() / 2,
            f"{sub.loc[i, 'score']:.3g}",
            va="center",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(outdir / f"fig5_top{int(topk):02d}_barh.png", dpi=300)
    plt.close()


def write_gene_mapped_table(diff_df, out_tsv, map_df):
    d = diff_df.copy()
    d["node_clean"] = clean_node_series(d["node"])
    m = map_df[["accession", "gene_symbol"]].drop_duplicates("accession")
    d = d.merge(m, left_on="node_clean", right_on="accession", how="left")

    d["gene_symbol"] = d["gene_symbol"].astype(str)
    d.loc[d["gene_symbol"].str.strip() == "C1orf39", "gene_symbol"] = "TOCA-1"

    d["node_display"] = d["gene_symbol"].replace("nan", np.nan).fillna(d["node"])
    d = d.sort_values("score", ascending=False)

    cols = [c for c in ["node_display", "score", "degree", "weighted_degree", "node", "node_clean", "gene_symbol"] if c in d.columns]
    out = d[cols].rename(columns={"node_display": "node"})
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep="\t", index=False)

    return int(d["gene_symbol"].notna().sum()), int(len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="inputs/pkl/nodes.final.tsv")
    ap.add_argument("--edges", default="inputs/pkl/edges.final.tsv")
    ap.add_argument("--diffusion", default="outputs/diffusion/diffusion_scores_alpha0.85.csv")
    ap.add_argument("--graph-pkl", default="inputs/pkl/ppi_subgraph.pkl")
    ap.add_argument("--outdir", default="outputs/figures")
    ap.add_argument("--max-hop", type=int, default=2)
    ap.add_argument("--max-hop-box", type=int, default=4)
    ap.add_argument("--top50", type=int, default=50)
    ap.add_argument("--topk-heatmap", type=int, default=50)
    ap.add_argument("--layout-seed", type=int, default=42)
    ap.add_argument("--write-gene-tsv", action="store_true")
    ap.add_argument("--gene-tsv-path", default="outputs/diffusion/diffusion_scores_gene.tsv")
    ap.add_argument("--uniprot-batch-size", type=int, default=500)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    outdir = ROOT / args.outdir
    ensure_outdir(outdir)

    nodes = pd.read_csv(ROOT / args.nodes, sep="\t", dtype=str)
    nodes["entry"] = nodes["entry"].astype(str)
    nodes["is_seed"] = pd.to_numeric(nodes.get("is_seed", 0), errors="coerce").fillna(0).astype(int)

    diff = pd.read_csv(ROOT / args.diffusion, dtype=str)
    diff["node"] = diff["node"].astype(str)
    diff["score"] = pd.to_numeric(diff["score"], errors="coerce")
    diff["degree"] = pd.to_numeric(diff.get("degree"), errors="coerce")
    diff["weighted_degree"] = pd.to_numeric(diff.get("weighted_degree"), errors="coerce")
    diff_valid = diff.dropna(subset=["score"]).copy()

    mapping = fetch_uniprot_names(clean_node_series(diff_valid["node"]), batch_size=args.uniprot_batch_size)
    gene_map, _ = build_display_maps(mapping)

    if args.write_gene_tsv:
        out_tsv = ROOT / args.gene_tsv_path
        m, t = write_gene_mapped_table(diff_valid, out_tsv, mapping)
        print(f"Wrote {out_tsv} mapped {m}/{t} ({m/t:.1%})")

    g = load_graph(ROOT / args.edges, ROOT / args.graph_pkl if args.graph_pkl else None)
    seeds = nodes.loc[nodes["is_seed"] == 1, "entry"].tolist()
    hops = multi_source_hops(g, seeds, cutoff=max(args.max_hop, args.max_hop_box))
    rank_map = build_rank_map(diff_valid)

    plot_hop_subgraph_structure_rank_shaded(g, nodes, hops, rank_map, outdir, args.max_hop, args.layout_seed)
    plot_rank_score_curve(diff_valid, outdir)
    plot_score_vs_hop_boxplot(diff_valid, hops, outdir, args.max_hop_box)

    if int(args.top50) > 0:
        plot_topk_induced_subgraph(g, nodes, diff_valid, gene_map, outdir, args.top50, args.layout_seed)

    plot_topk_barh(diff_valid, nodes, gene_map, outdir, topk=25)


if __name__ == "__main__":
    main()
