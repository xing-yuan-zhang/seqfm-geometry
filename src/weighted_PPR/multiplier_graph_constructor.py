from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import networkx as nx
import pickle


def clean_nan_for_graphml(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if pd.isna(v):
            out[k] = ""
        else:
            out[k] = v
    return out


def _to_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _to_int(x, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if x is None or (hasattr(pd, "isna") and pd.isna(x)):
        return False
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def compute_seed_weight_from_annotations(row: pd.Series) -> float:
    """
    Conservative defaults:
      - LLPS prior: ×1.5 (if is_LLPS_any==1)
      - SH3: ×1.2
      - PRD: ×1.2
      - SH3-related ELM motifs: +10% per motif, capped at +50%
      - length >= 800: ×1.1
    """
    w = 1.0

    if _to_int(row.get("is_LLPS_any", 0), 0) == 1:
        w *= 1.5

    if _to_bool(row.get("has_SH3", False)):
        w *= 1.2
    if _to_bool(row.get("has_PRD", False)):
        w *= 1.2

    c = row.get("elm_sh3_related", None)
    if c is None or (hasattr(pd, "isna") and pd.isna(c)):
        c = _to_float(row.get("elm_sh3_related_x", 0.0), 0.0) + _to_float(row.get("elm_sh3_related_y", 0.0), 0.0)
    else:
        c = _to_float(c, 0.0)

    if c > 0:
        w *= (1.0 + min(0.5, 0.1 * c))

    L = _to_float(row.get("length", 0.0), 0.0)
    if L >= 800:
        w *= 1.1

    return float(w)


def compute_node_prior_mult_from_annotations(row: pd.Series) -> float:
    """
    Annotation-aware node prior multiplier.
    This is intentionally milder than seed_weight defaults:
      - LLPS prior: ×1.2 (if is_LLPS_any==1)
      - SH3: ×1.1
      - PRD: ×1.1
      - SH3-related ELM motifs: +10% per motif, capped at +50%
      - length >= 800: ×1.05
    """
    m = 1.0

    if _to_int(row.get("is_LLPS_any", 0), 0) == 1:
        m *= 1.2

    if _to_bool(row.get("has_SH3", False)):
        m *= 1.1

    if _to_bool(row.get("has_PRD", False)):
        m *= 1.1

    k = _to_int(row.get("elm_sh3_related", 0), 0)
    if k > 0:
        m *= (1.0 + 0.10 * min(k, 5))

    L = _to_float(row.get("length", 0.0), 0.0)
    if L >= 800:
        m *= 1.05

    return float(m)


def choose_seed_component(G: nx.Graph, seeds: set[str]) -> set[str] | None:
    seeds_in_graph = seeds.intersection(G.nodes)
    if not seeds_in_graph:
        return None

    best_comp = None
    best_count = -1
    best_size = -1

    for comp in nx.connected_components(G):
        comp_set = set(comp)
        c = len(comp_set.intersection(seeds_in_graph))
        if c > best_count or (c == best_count and len(comp_set) > best_size):
            best_count = c
            best_size = len(comp_set)
            best_comp = comp_set

    return best_comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="inputs/ppi/subgraph/subgraph_edges.tsv")
    ap.add_argument("--nodes", default="inputs/ppi/subgraph/subgraph_node_attributes.tsv")
    ap.add_argument("--seeds", default="inputs/ppi/subgraph/subgraph_nodes.tsv",
                    help="need entry and is_seed columns from subgraph_nodes.tsv")
    ap.add_argument("--outdir", default="inputs/pkl")
    ap.add_argument("--keep", choices=["none", "largest", "seed"], default="seed",
                    help="none=connectivity filtering, no cropping; largest=retain the largest component; seed=retain the component containing the most seeds")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    edges_path = str(ROOT / args.edges)
    nodes_path = Path(ROOT / args.nodes)
    seeds_path = Path(ROOT / args.seeds)
    outdir = Path(ROOT / args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    e = pd.read_csv(edges_path, sep="\t")
    n = pd.read_csv(nodes_path, sep="\t")

    for c in ["entry_a", "entry_b", "weight"]:
        if c not in e.columns:
            raise ValueError(f"{edges_path} need column {c}")
    if "entry" not in n.columns:
        raise ValueError(f"{nodes_path} need entry column")

    seeds_df = pd.read_csv(seeds_path, sep="\t")
    if not {"entry", "is_seed"}.issubset(set(seeds_df.columns)):
        raise ValueError(f"{seeds_path} need entry and is_seed columns")
    seeds = set(seeds_df.loc[seeds_df["is_seed"].astype(int) == 1, "entry"].astype(str).tolist())

    seed_flag_map = dict(
        zip(
            seeds_df["entry"].astype(str).tolist(),
            seeds_df["is_seed"].astype(int).tolist(),
        )
    )
    if "is_seed" in n.columns:
        n["is_seed"] = n["entry"].astype(str).map(seed_flag_map).fillna(n["is_seed"]).astype(int)
    else:
        n["is_seed"] = n["entry"].astype(str).map(seed_flag_map).fillna(0).astype(int)

    n["seed_weight"] = 0.0
    seed_mask = n["is_seed"].astype(int) == 1
    if seed_mask.any():
        n.loc[seed_mask, "seed_weight"] = n.loc[seed_mask].apply(compute_seed_weight_from_annotations, axis=1)

    n["node_prior_mult"] = n.apply(compute_node_prior_mult_from_annotations, axis=1)


    G = nx.Graph()

    for _, r in n.iterrows():
        d = r.to_dict()
        node = str(d.pop("entry"))
        d = clean_nan_for_graphml(d)
        G.add_node(node, **d)

    for _, r in e.iterrows():
        a = str(r["entry_a"])
        b = str(r["entry_b"])
        attrs = {
            "weight": float(r["weight"]),
            "is_string": int(r.get("is_string", 0)) if not pd.isna(r.get("is_string", 0)) else 0,
            "is_biogrid": int(r.get("is_biogrid", 0)) if not pd.isna(r.get("is_biogrid", 0)) else 0,
        }
        if "string_score" in e.columns and not pd.isna(r.get("string_score", pd.NA)):
            attrs["string_score"] = float(r["string_score"])
        G.add_edge(a, b, **attrs)

    num_cc = nx.number_connected_components(G)
    cc_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    largest_cc = cc_sizes[0] if cc_sizes else 0

    seeds_in_graph = len(seeds.intersection(G.nodes))

    stats_lines = []
    stats_lines.append(f"graph BEFORE filtering: nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    stats_lines.append(f"connected components={num_cc:,}; largest_component_size={largest_cc:,}")
    stats_lines.append(f"seeds total={len(seeds):,}; seeds_in_graph={seeds_in_graph:,}")

    kept_nodes = None
    if args.keep == "largest" and G.number_of_nodes() > 0:
        comp = max(nx.connected_components(G), key=len)
        kept_nodes = set(comp)
        G = G.subgraph(kept_nodes).copy()
        stats_lines.append(f"KEEP=largest: kept_nodes={len(kept_nodes):,}")
    elif args.keep == "seed":
        comp = choose_seed_component(G, seeds)
        if comp is not None:
            kept_nodes = comp
            G = G.subgraph(kept_nodes).copy()
            stats_lines.append(f"KEEP=seed: kept_nodes={len(kept_nodes):,} (seed-component)")
        else:
            stats_lines.append("KEEP=seed: no seeds found in graph; no filtering applied")

    num_cc2 = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
    cc_sizes2 = sorted([len(c) for c in nx.connected_components(G)], reverse=True) if G.number_of_nodes() > 0 else []
    largest_cc2 = cc_sizes2[0] if cc_sizes2 else 0
    stats_lines.append(f"graph AFTER filtering: nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    stats_lines.append(f"connected components={num_cc2:,}; largest_component_size={largest_cc2:,}")
    stats_lines.append(f"seeds_in_graph_after={len(seeds.intersection(G.nodes)):,}")

    nx.write_graphml(G, outdir / "ppi_subgraph.graphml")
    with open(outdir / "ppi_subgraph.pkl", "wb") as f:
        pickle.dump(G, f)


    final_nodes = pd.DataFrame({"entry": list(G.nodes)})
    final_nodes = final_nodes.merge(n, on="entry", how="left")
    final_edges = nx.to_pandas_edgelist(G)
    final_edges = final_edges.rename(columns={"source": "entry_a", "target": "entry_b"})
    final_nodes.to_csv(outdir / "nodes.final.tsv", sep="\t", index=False)

    seed_weights = final_nodes[["entry", "is_seed", "seed_weight"]].copy()
    seed_weights.to_csv(outdir / "seed_weights.tsv", sep="\t", index=False)

    final_edges.to_csv(outdir / "edges.final.tsv", sep="\t", index=False)

    (outdir / "graph.stats.txt").write_text("\n".join(stats_lines) + "\n")

    print(f"graph nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    print(f"wrote: {outdir/'ppi_subgraph.pkl'}")
    print(f"wrote: {outdir/'graph.stats.txt'}")


if __name__ == "__main__":
    main()
