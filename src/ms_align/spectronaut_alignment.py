import argparse
import re
from pathlib import Path
from typing import List, Set, Tuple, Optional

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


UNIPROT_RE = re.compile(
    r"^(?:[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9]|A0A[A-Z0-9]{7,10})$"
)
SPLIT_RE = re.compile(r"[;,\s]+")


def normalize_uniprot(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    if not x:
        return ""
    return x.split("-")[0]


def extract_uniprots_from_cell(cell) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    out = []
    for p in SPLIT_RE.split(s):
        p = normalize_uniprot(p)
        if p and UNIPROT_RE.match(p):
            out.append(p)
    return out


def detect_uniprot_column(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"--ms_id_col '{user_col}' err: {list(df.columns)}")
        return user_col

    best, best_score = None, -1.0
    for col in df.columns:
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        hit = s.map(lambda v: 1.0 if extract_uniprots_from_cell(v) else 0.0).mean()
        bonus = 0.05 if str(col).lower() in {
            "entry",
            "accession",
            "uniprot",
            "uniprot_id",
            "protein",
            "pg.proteinaccessions",
        } else 0.0
        score = hit + bonus
        if score > best_score:
            best, best_score = col, score

    if best is None or best_score < 0.05:
        raise ValueError("accession error")
    return best


def load_ms_uniprots(ms_path: str, ms_id_col: Optional[str]) -> Set[str]:
    df = pd.read_csv(ms_path, sep="\t", low_memory=False)
    col = detect_uniprot_column(df, ms_id_col)
    out = set()
    for v in df[col].dropna():
        out.update(extract_uniprots_from_cell(v))
    return out


def load_edges(path: str, cols=("entry_a", "entry_b")) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"need column '{c}', have column {list(df.columns)}")
    df[cols[0]] = df[cols[0]].astype(str).map(normalize_uniprot)
    df[cols[1]] = df[cols[1]].astype(str).map(normalize_uniprot)
    df = df[df[cols[0]].str.len() > 0]
    df = df[df[cols[1]].str.len() > 0]
    return df


def induced_subgraph_edges(edges: pd.DataFrame, nodes: Set[str], a="entry_a", b="entry_b"):
    return edges.loc[edges[a].isin(nodes) & edges[b].isin(nodes)].copy()


def to_undirected_edge_set(edges: pd.DataFrame, a="entry_a", b="entry_b") -> Set[Tuple[str, str]]:
    s = set()
    for u, v in zip(edges[a], edges[b]):
        if not u or not v or u == v:
            continue
        s.add(tuple(sorted((u, v))))
    return s


def build_graph_from_edges(edges: pd.DataFrame, a="entry_a", b="entry_b", weight_col="weight") -> nx.Graph:
    G = nx.Graph()
    if weight_col and weight_col in edges.columns:
        for u, v, w in zip(edges[a], edges[b], edges[weight_col]):
            if u != v:
                G.add_edge(u, v, weight=float(w))
    else:
        for u, v in zip(edges[a], edges[b]):
            if u != v:
                G.add_edge(u, v)
    return G


def take_plot_subgraph(G: nx.Graph, max_nodes: int):
    if G.number_of_nodes() <= max_nodes:
        return G
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    H = G.subgraph(comps[0]).copy()
    if H.number_of_nodes() <= max_nodes:
        return H
    deg = sorted(H.degree, key=lambda x: x[1], reverse=True)
    keep = {n for n, _ in deg[:max_nodes]}
    return H.subgraph(keep).copy()


def draw_ms_graph(G: nx.Graph, out_png: str, max_nodes: int, seed: int):
    H = take_plot_subgraph(G, max_nodes)
    pos = nx.spring_layout(H, seed=seed)
    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(H, pos, alpha=0.25, width=0.7)
    nx.draw_networkx_nodes(H, pos, node_size=18)
    plt.title(f"MS embedding on STRING (nodes={H.number_of_nodes()}, edges={H.number_of_edges()})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def draw_top_pf_graph(edges: pd.DataFrame, pf_df: pd.DataFrame, out_png: str, topk: int, seed: int):
    top_nodes = pf_df.head(topk)["uniprot"].tolist()
    sub_edges = induced_subgraph_edges(edges, set(top_nodes))
    G = build_graph_from_edges(sub_edges)

    if G.number_of_nodes() == 0:
        return

    pos = nx.spring_layout(G, seed=seed, k=0.08)
    for n in pos:
        pos[n] *= 0.6

    plt.figure(figsize=(9, 9))
    nx.draw_networkx_edges(G, pos, edge_color="#c0c0c0", alpha=0.18, width=0.8)
    nx.draw_networkx_nodes(G, pos, node_size=380, node_color="#4f83cc", linewidths=0)
    nx.draw_networkx_labels(G, pos, font_size=10)

    plt.title(f"Top-{topk} Pillar/Flat ratio (STRING)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def compute_pf_ratio(ms_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(ms_tsv, sep="\t", low_memory=False)

    q_cols = [c for c in df.columns if c.endswith(".PG.Quantity")]
    if len(q_cols) < 2:
        raise ValueError("need both Flat and Pillar quantity columns")

    flat_col = [c for c in q_cols if "Flat" in c][0]
    pillar_col = [c for c in q_cols if "Pillar" in c][0]

    df["Flat"] = pd.to_numeric(df[flat_col], errors="coerce")
    df["Pillar"] = pd.to_numeric(df[pillar_col], errors="coerce")
    df = df.dropna(subset=["Flat", "Pillar"])
    df = df[(df["Flat"] > 0) & (df["Pillar"] > 0)]

    df["PF_ratio"] = df["Pillar"] / df["Flat"]

    df["uniprot"] = df["PG.ProteinAccessions"].map(extract_uniprots_from_cell)
    df = df.explode("uniprot")
    df = df[df["uniprot"].str.len() > 0]

    out = (
        df.groupby(["uniprot", "PG.Genes"], as_index=False)["PF_ratio"]
        .mean()
        .sort_values("PF_ratio", ascending=False)
    )
    return out


def draw_overlay(
    G_ms: nx.Graph,
    G_diff: nx.Graph,
    overlap_nodes: Set[str],
    overlap_edges: Set[Tuple[str, str]],
    out_png: str,
    max_nodes: int,
    seed: int,
):
    U = nx.Graph()
    U.add_nodes_from(G_ms.nodes())
    U.add_nodes_from(G_diff.nodes())
    U.add_edges_from(G_ms.edges())
    U.add_edges_from(G_diff.edges())

    H = take_plot_subgraph(U, max_nodes)
    pos = nx.spring_layout(H, seed=seed)

    ms_nodes = set(G_ms.nodes())
    diff_nodes = set(G_diff.nodes())

    nodes_only_ms = [n for n in H.nodes() if n in ms_nodes and n not in diff_nodes]
    nodes_only_diff = [n for n in H.nodes() if n in diff_nodes and n not in ms_nodes]
    nodes_overlap = [n for n in H.nodes() if n in overlap_nodes]

    ms_es = {tuple(sorted(e)) for e in G_ms.edges()}
    diff_es = {tuple(sorted(e)) for e in G_diff.edges()}
    H_es = {tuple(sorted(e)) for e in H.edges()}

    edges_overlap = [e for e in H_es if e in overlap_edges]
    edges_only_ms = [e for e in H_es if e in ms_es and e not in diff_es]
    edges_only_diff = [e for e in H_es if e in diff_es and e not in ms_es]

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(H, pos, edgelist=edges_only_ms, alpha=0.18, width=0.8)
    nx.draw_networkx_edges(H, pos, edgelist=edges_only_diff, alpha=0.18, width=0.8, style="dashed")
    nx.draw_networkx_edges(H, pos, edgelist=edges_overlap, alpha=0.85, width=1.6)

    nx.draw_networkx_nodes(H, pos, nodelist=nodes_only_ms, node_size=18, alpha=0.65)
    nx.draw_networkx_nodes(H, pos, nodelist=nodes_only_diff, node_size=18, alpha=0.65)
    nx.draw_networkx_nodes(H, pos, nodelist=nodes_overlap, node_size=42, alpha=0.95)

    plt.title(
        "Overlay: MS vs diffusion subgraph\n"
        f"Union nodes={H.number_of_nodes()} edges={H.number_of_edges()} | "
        f"Overlap nodes={len(overlap_nodes)} edges={len(overlap_edges)}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms_tsv", required=True)
    ap.add_argument("--string_edges", required=True)
    ap.add_argument("--diff_edges", required=True)
    ap.add_argument("--ms_id_col", default=None)
    ap.add_argument("--outdir", default="ms_vs_diffusion_out")
    ap.add_argument("--max_nodes_ms", type=int, default=800)
    ap.add_argument("--max_nodes_overlay", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    ms_ids = load_ms_uniprots(str(ROOT / args.ms_tsv), args.ms_id_col)
    if not ms_ids:
        raise RuntimeError("accession not found")

    string_df = load_edges(str(ROOT / args.string_edges))
    ms_edges_df = induced_subgraph_edges(string_df, ms_ids)

    diff_df = load_edges(str(ROOT / args.diff_edges))

    G_ms = build_graph_from_edges(ms_edges_df)
    G_diff = build_graph_from_edges(diff_df)

    ms_nodes = set(G_ms.nodes())
    diff_nodes = set(G_diff.nodes())
    overlap_nodes = ms_nodes & diff_nodes

    ms_es = to_undirected_edge_set(ms_edges_df)
    diff_es = to_undirected_edge_set(diff_df)
    overlap_edges = ms_es & diff_es

    with open(outdir / "overlap_stats.txt", "w") as f:
        f.write(f"MS proteins extracted: {len(ms_ids)}\n")
        f.write(f"MS-induced graph nodes: {len(ms_nodes)} edges: {G_ms.number_of_edges()}\n")
        f.write(f"Diffusion graph nodes: {len(diff_nodes)} edges: {G_diff.number_of_edges()}\n")
        f.write(f"Overlap nodes: {len(overlap_nodes)}\n")
        f.write(f"Overlap edges: {len(overlap_edges)}\n")

    pd.Series(sorted(overlap_nodes), name="uniprot").to_csv(outdir / "overlap_nodes.tsv", sep="\t", index=False)
    pd.DataFrame(sorted(list(overlap_edges)), columns=["entry_a", "entry_b"]).to_csv(
        outdir / "overlap_edges.tsv", sep="\t", index=False
    )

    draw_ms_graph(G_ms, str(outdir / "ms_induced_graph.png"), args.max_nodes_ms, args.seed)
    draw_overlay(
        G_ms,
        G_diff,
        overlap_nodes,
        overlap_edges,
        str(outdir / "overlay_ms_vs_diffusion.png"),
        args.max_nodes_overlay,
        args.seed,
    )

    pf_df = compute_pf_ratio(ROOT / args.ms_tsv)
    pf_df.to_csv(outdir / "pf_ratio_ranked.tsv", sep="\t", index=False)

    draw_top_pf_graph(string_df, pf_df, str(outdir / "top_pf_ratio.png"), 100, args.seed)
    draw_top_pf_graph(
        string_df,
        pf_df,
        str(outdir / "top50_pf_ratio_string.png"),
        topk=50,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
