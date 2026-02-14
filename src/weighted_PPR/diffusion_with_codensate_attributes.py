from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

import networkx as nx
import pandas as pd
import pickle

try:
    import yaml
except ImportError as e:
    raise SystemExit() from e


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("candidate_diffusion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


@dataclass
class SeedItem:
    query: str
    weight: float = 1.0
    role: str = ""
    evidence: List[str] = field(default_factory=list)


@dataclass
class ControlItem:
    query: str
    role: str = ""
    evidence: List[str] = field(default_factory=list)


def load_graph(path: str):
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def load_seeds_yaml(path: str) -> Tuple[List[SeedItem], List[ControlItem]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    seeds_raw = obj.get("seeds", []) or []
    ctrls_raw = obj.get("negative_controls", []) or []

    seeds: List[SeedItem] = []
    for x in seeds_raw:
        seeds.append(
            SeedItem(
                query=str(x.get("query", "")).strip(),
                weight=float(x.get("weight", 1.0)),
                role=str(x.get("role", "")).strip(),
                evidence=list(x.get("evidence", []) or []),
            )
        )

    ctrls: List[ControlItem] = []
    for x in ctrls_raw:
        ctrls.append(
            ControlItem(
                query=str(x.get("query", "")).strip(),
                role=str(x.get("role", "")).strip(),
                evidence=list(x.get("evidence", []) or []),
            )
        )

    return seeds, ctrls


def build_personalization(
    G: nx.Graph,
    seeds: List[SeedItem],
    exclude_roles: Set[str],
    logger: logging.Logger,
    seed_weight_col: Optional[str] = "seed_weight",
    seed_weight_clip: Tuple[float, float] = (0.1, 10.0),
) -> Tuple[Dict[str, float], List[str], List[str]]:
    missing: List[str] = []
    active: List[str] = []
    weights: Dict[str, float] = {}

    def _get_seed_multiplier(node: str) -> float:
        if not seed_weight_col:
            return 1.0
        try:
            v = G.nodes[node].get(seed_weight_col, None)
        except Exception:
            return 1.0
        if v is None:
            return 1.0
        try:
            fv = float(v)
        except Exception:
            return 1.0
        if not (fv > 0):
            return 1.0
        lo, hi = seed_weight_clip
        return max(lo, min(hi, fv))

    for s in seeds:
        if not s.query:
            continue
        if s.role in exclude_roles:
            missing.append(s.query)
            continue
        if s.query not in G:
            missing.append(s.query)
            continue
        w0 = float(s.weight) if s.weight is not None else 1.0
        if w0 <= 0:
            missing.append(s.query)
            continue
        mult = _get_seed_multiplier(s.query)
        w = w0 * mult
        weights[s.query] = weights.get(s.query, 0.0) + w
        active.append(s.query)

    if not weights:
        raise ValueError()

    total = sum(weights.values())
    personalization = {k: v / total for k, v in weights.items()}

    logger.info(f"seeds used: {len(personalization)}")
    if seed_weight_col:
        try:
            mults = {k: _get_seed_multiplier(k) for k in personalization.keys()}
            show = list(sorted(mults.items(), key=lambda x: x[1], reverse=True))[:10]
            logger.info(f"seed weight col='{seed_weight_col}' (clip={seed_weight_clip}); top multipliers: {show}")
        except Exception:
            pass
    if missing:
        logger.warning(
            f"seeds missing or excluded: {len(missing)} -> {missing[:20]}{'...' if len(missing)>20 else ''}"
        )

    return personalization, sorted(set(active)), sorted(set(missing))



def _as_bool(v) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def compute_node_multipliers(
    G: nx.Graph,
    clip: Tuple[float, float] = (0.2, 5.0),
    llps_mult: float = 1.2,
    sh3_mult: float = 1.1,
    prd_mult: float = 1.1,
    motif_step: float = 0.10,
    motif_cap: int = 5,
    length_threshold: int = 800,
    length_mult: float = 1.05,
    llps_col: str = "is_LLPS_any",
    sh3_col: str = "has_SH3",
    prd_col: str = "has_PRD",
    motif_col: str = "elm_sh3_related",
    length_col: str = "length",
) -> Dict[str, float]:
    """
    Incorporating node annotations into transition weights.
    Implement a target-node multiplier m(v) and define directed edge weights:
        w'(u->v) = w(u,v) * m(v)
    Biases random walks to flow preferentially into nodes with LLPS/SH3/PRD evidence.
    """
    lo, hi = clip
    mults: Dict[str, float] = {}
    for n, d in G.nodes(data=True):
        m = 1.0
        try:
            if int(float(d.get(llps_col, 0))) == 1:
                m *= llps_mult
        except Exception:
            pass
        if _as_bool(d.get(sh3_col, False)):
            m *= sh3_mult
        if _as_bool(d.get(prd_col, False)):
            m *= prd_mult
        # motif bonus
        k = 0
        try:
            k = int(float(d.get(motif_col, 0)))
        except Exception:
            k = 0
        if k > 0:
            m *= (1.0 + motif_step * min(k, motif_cap))
        # very weak length bonus
        try:
            L = float(d.get(length_col, 0))
            if L >= length_threshold:
                m *= length_mult
        except Exception:
            pass

        if not (m > 0):
            m = 1.0
        m = max(lo, min(hi, m))
        mults[str(n)] = float(m)
    return mults


def build_transition_graph_with_node_prior(
    G: nx.Graph,
    node_mult: Dict[str, float],
    base_weight_attr: Optional[str],
    out_weight_attr: str = "w_adj",
) -> nx.DiGraph:
    """
    Convert an undirected graph G into a directed graph H with adjusted edge weights:
        H[u->v].w_adj = G[u,v].base_weight * node_mult[v]
    If base_weight_attr is None, base_weight is 1.0.
    """
    H = nx.DiGraph()
    # copy nodes + attributes
    for n, d in G.nodes(data=True):
        H.add_node(str(n), **d)
    for u, v, ed in G.edges(data=True):
        u = str(u); v = str(v)
        if base_weight_attr:
            try:
                w = float(ed.get(base_weight_attr, 1.0))
            except Exception:
                w = 1.0
        else:
            w = 1.0
        mu = float(node_mult.get(u, 1.0))
        mv = float(node_mult.get(v, 1.0))
        # u -> v depends on target v
        H.add_edge(u, v, **{out_weight_attr: w * mv})
        H.add_edge(v, u, **{out_weight_attr: w * mu})
    return H


def run_ppr(
    G: nx.Graph,
    alpha: float,
    personalization: Dict[str, float],
    weight_attr: Optional[str],
    max_iter: int,
    tol: float,
    logger: logging.Logger,
) -> Dict[str, float]:
    logger.info(
        f"Running PPR with alpha={alpha:.3f}, restart={1-alpha:.3f}, weight_attr={weight_attr}, max_iter={max_iter}, tol={tol}"
    )
    try:
        scores = nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            weight=weight_attr if weight_attr else None,
            max_iter=max_iter,
            tol=tol,
        )
    except nx.PowerIterationFailedConvergence as e:
        raise RuntimeError(
            f"PageRank did not converge (alpha={alpha}, max_iter={max_iter}, tol={tol}). Try increasing --max_iter or relaxing --tol."
        ) from e
    return scores


def add_degree_features(G: nx.Graph, df: pd.DataFrame, weight_attr: Optional[str]) -> pd.DataFrame:
    deg = dict(G.degree())
    df["degree"] = df["node"].map(deg).fillna(0).astype(int)

    if weight_attr:
        try:
            wdeg = dict(G.degree(weight=weight_attr))
            df["weighted_degree"] = df["node"].map(wdeg).fillna(0.0)
        except Exception:
            pass
    return df


def maybe_merge_node_attributes(
    df: pd.DataFrame,
    node_attr_path: Optional[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    if not node_attr_path:
        return df
    if not os.path.exists(node_attr_path):
        logger.warning(f"Node attributes file not found: {node_attr_path}")
        return df

    na = pd.read_csv(node_attr_path, sep="\t")
    if "entry" in na.columns and "node" not in na.columns:
        na = na.rename(columns={"entry": "node"})
    if "node" not in na.columns:
        logger.warning(f"Missing 'entry' or 'node' column: {node_attr_path}")
        return df

    merged = df.merge(na, on="node", how="left")
    logger.info(f"Merged node attributes from {node_attr_path}")
    return merged


def compute_qc(
    df: pd.DataFrame,
    seed_nodes: Set[str],
    negative_nodes: Set[str],
    topk: int = 50,
    llps_col: str = "is_LLPS_any",
) -> Dict:
    df2 = df.copy()
    df2["rank"] = range(1, len(df2) + 1)

    seed_df = df2[df2["node"].isin(seed_nodes)]
    neg_df = df2[df2["node"].isin(negative_nodes)]

    qc = {
        "n_nodes": int(len(df2)),
        "topk": int(topk),
        "n_seeds": int(len(seed_nodes)),
        "n_neg_controls": int(len(negative_nodes)),
        "seeds_in_topk": int((seed_df["rank"] <= topk).sum()) if len(seed_df) else 0,
        "seed_rank_mean": float(seed_df["rank"].mean()) if len(seed_df) else None,
        "seed_rank_median": float(seed_df["rank"].median()) if len(seed_df) else None,
        "neg_rank_mean": float(neg_df["rank"].mean()) if len(neg_df) else None,
        "neg_rank_median": float(neg_df["rank"].median()) if len(neg_df) else None,
        "top10_nodes": df2.head(10)[["node", "score"]].to_dict(orient="records"),
    }

    if len(seed_df):
        qc["seed_ranks"] = seed_df.sort_values("rank")[["node", "rank", "score"]].to_dict(orient="records")
    else:
        qc["seed_ranks"] = []

    if len(neg_df):
        qc["neg_control_ranks"] = neg_df.sort_values("rank")[["node", "rank", "score"]].to_dict(orient="records")
    else:
        qc["neg_control_ranks"] = []

    if llps_col in df2.columns:
        top = df2.head(topk)
        base = df2[llps_col].fillna(0).astype(int)
        topv = top[llps_col].fillna(0).astype(int)
        qc["llps_col"] = llps_col
        qc["llps_base_rate"] = float(base.mean()) if len(base) else None
        qc["llps_in_topk"] = int(topv.sum())
        qc["llps_rate_in_topk"] = float(topv.mean()) if len(topv) else None

    return qc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, help="Path to NetworkX graph gpickle/pkl")
    p.add_argument("--seeds", required=True, help="Path to seeds.yaml (queries must match graph node IDs)")
    p.add_argument("--outdir", required=True, help="Output directory, e.g., outputs/diffusion")
    p.add_argument("--alpha", nargs="+", type=float, default=[0.7])
    p.add_argument("--topn", type=int, default=100)
    p.add_argument("--weight_attr", type=str, default="weight")
    p.add_argument("--qc_topk", type=int, default=50)
    p.add_argument("--exclude_seed_roles", nargs="*", default=[])
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--node_attributes", type=str, default="")
    p.add_argument("--llps_col", type=str, default="is_LLPS_any")
    p.add_argument("--seed_weight_col", type=str, default="seed_weight",
                   help="Graph node attribute used to up-weight seeds. Set '' to disable.")
    p.add_argument("--seed_weight_clip_min", type=float, default=0.1,
                   help="Clip lower bound for seed_weight multiplier (safety).")
    p.add_argument("--seed_weight_clip_max", type=float, default=10.0,
                   help="Clip upper bound for seed_weight multiplier (safety).")
    p.add_argument("--node_prior_mode", choices=["none", "target"], default="target",
                   help="Incorporate node annotations into transition weights. "
                        "'target' biases walks to flow into nodes with LLPS/SH3/PRD evidence; 'none' disables.")
    p.add_argument("--node_prior_col", type=str, default="",
                   help="Optional precomputed node attribute to use as multiplier (e.g., 'node_prior_mult'). "
                        "If empty, multipliers are computed from annotation columns.")
    p.add_argument("--node_prior_clip_min", type=float, default=0.2)
    p.add_argument("--node_prior_clip_max", type=float, default=5.0)
    p.add_argument("--node_prior_llps_mult", type=float, default=1.2)
    p.add_argument("--node_prior_sh3_mult", type=float, default=1.1)
    p.add_argument("--node_prior_prd_mult", type=float, default=1.1)
    p.add_argument("--node_prior_motif_step", type=float, default=0.10)
    p.add_argument("--node_prior_motif_cap", type=int, default=5)
    p.add_argument("--node_prior_length_threshold", type=int, default=800)
    p.add_argument("--node_prior_length_mult", type=float, default=1.05)
    p.add_argument("--node_prior_llps_col", type=str, default="is_LLPS_any")
    p.add_argument("--node_prior_sh3_col", type=str, default="has_SH3")
    p.add_argument("--node_prior_prd_col", type=str, default="has_PRD")
    p.add_argument("--node_prior_motif_col", type=str, default="elm_sh3_related")
    p.add_argument("--node_prior_length_col", type=str, default="length")
    return p.parse_args()


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]

    args = parse_args()
    outdir = str(ROOT / args.outdir)
    os.makedirs(outdir, exist_ok=True)

    log_path = os.path.join(outdir, "diffusion.log")
    logger = setup_logger(log_path)

    logger.info(f"Graph: {args.graph}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Outdir: {outdir}")

    G = load_graph(str(ROOT / args.graph))
    logger.info(f"Loaded graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}, directed={G.is_directed()}")

    seeds, ctrls = load_seeds_yaml(str(ROOT / args.seeds))

    exclude_roles = set(args.exclude_seed_roles or [])
    personalization, active_seed_nodes, missing_seed_nodes = build_personalization(
        G,
        seeds,
        exclude_roles,
        logger,
        seed_weight_col=(args.seed_weight_col.strip() if args.seed_weight_col is not None else None) or None,
        seed_weight_clip=(args.seed_weight_clip_min, args.seed_weight_clip_max),
    )

    seed_set = set(active_seed_nodes)
    neg_set = set([c.query for c in ctrls if c.query in G])

    node_attr_path = args.node_attributes.strip()
    if not node_attr_path:
        inferred = os.path.join(os.path.dirname(os.path.abspath(args.graph)), "nodes.final.tsv")
        if os.path.exists(inferred):
            node_attr_path = inferred

    config = {
        "graph": args.graph,
        "seeds_yaml": args.seeds,
        "alphas": args.alpha,
        "topn": args.topn,
        "weight_attr": args.weight_attr if args.weight_attr else None,
        "exclude_seed_roles": sorted(list(exclude_roles)),
        "active_seeds": active_seed_nodes,
        "missing_or_excluded_seeds": missing_seed_nodes,
        "negative_controls_in_graph": sorted(list(neg_set)),
        "max_iter": args.max_iter,
        "tol": args.tol,
        "node_attributes": node_attr_path if node_attr_path else None,
        "llps_col": args.llps_col,
        "seed_weight_col": (args.seed_weight_col.strip() if args.seed_weight_col is not None else None) or None,
        "seed_weight_clip_min": args.seed_weight_clip_min,
        "seed_weight_clip_max": args.seed_weight_clip_max,
        "node_prior_mode": args.node_prior_mode,
        "node_prior_col": (args.node_prior_col or "").strip() or None,
        "node_prior_clip_min": args.node_prior_clip_min,
        "node_prior_clip_max": args.node_prior_clip_max,
        "node_prior_llps_mult": args.node_prior_llps_mult,
        "node_prior_sh3_mult": args.node_prior_sh3_mult,
        "node_prior_prd_mult": args.node_prior_prd_mult,
        "node_prior_motif_step": args.node_prior_motif_step,
        "node_prior_motif_cap": args.node_prior_motif_cap,
        "node_prior_length_threshold": args.node_prior_length_threshold,
        "node_prior_length_mult": args.node_prior_length_mult,
        "node_prior_llps_col": args.node_prior_llps_col,
        "node_prior_sh3_col": args.node_prior_sh3_col,
        "node_prior_prd_col": args.node_prior_prd_col,
        "node_prior_motif_col": args.node_prior_motif_col,
        "node_prior_length_col": args.node_prior_length_col,
    }
    with open(os.path.join(outdir, "diffusion_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    weight_attr = args.weight_attr.strip() if args.weight_attr else None
    if weight_attr == "":
        weight_attr = None

    diffusion_graph = G
    diffusion_weight_attr = weight_attr

    if args.node_prior_mode and args.node_prior_mode != "none":
        # Build per-node multipliers m(v), then define directed edge weights:
        # w'(u->v) = w(u,v) * m(v)
        col = (args.node_prior_col or "").strip()
        if col:
            node_mult = {}
            for n in diffusion_graph.nodes():
                v = diffusion_graph.nodes[n].get(col, 1.0)
                try:
                    fv = float(v)
                except Exception:
                    fv = 1.0
                node_mult[str(n)] = max(args.node_prior_clip_min, min(args.node_prior_clip_max, fv if fv > 0 else 1.0))
        else:
            node_mult = compute_node_multipliers(
                diffusion_graph,
                clip=(args.node_prior_clip_min, args.node_prior_clip_max),
                llps_mult=args.node_prior_llps_mult,
                sh3_mult=args.node_prior_sh3_mult,
                prd_mult=args.node_prior_prd_mult,
                motif_step=args.node_prior_motif_step,
                motif_cap=args.node_prior_motif_cap,
                length_threshold=args.node_prior_length_threshold,
                length_mult=args.node_prior_length_mult,
                llps_col=args.node_prior_llps_col,
                sh3_col=args.node_prior_sh3_col,
                prd_col=args.node_prior_prd_col,
                motif_col=args.node_prior_motif_col,
                length_col=args.node_prior_length_col,
            )

        top_mult = sorted(node_mult.items(), key=lambda kv: kv[1], reverse=True)[:20]
        logger.info(
            "Node multipliers enabled. Top multipliers: "
            + ", ".join([f"{n}:{m:.3g}" for n, m in top_mult[:10]])
        )

        diffusion_graph = build_transition_graph_with_node_prior(
            diffusion_graph,
            node_mult=node_mult,
            base_weight_attr=weight_attr,
            out_weight_attr="w_adj",
        )
        diffusion_weight_attr = "w_adj"

    for a in args.alpha:
        if not (0.0 < a < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {a}")

        scores = run_ppr(
            diffusion_graph,
            alpha=a,
            personalization=personalization,
            weight_attr=diffusion_weight_attr,
            max_iter=args.max_iter,
            tol=args.tol,
            logger=logger,
        )

        df = pd.DataFrame({"node": list(scores.keys()), "score": list(scores.values())})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df = add_degree_features(G, df, weight_attr=weight_attr)
        df = maybe_merge_node_attributes(df, node_attr_path, logger)

        full_path = os.path.join(outdir, f"diffusion_scores_alpha{a:.2f}.csv")
        df.to_csv(full_path, index=False)

        cand = df[~df["node"].isin(seed_set)].copy().reset_index(drop=True)
        cand["rank_excluding_seeds"] = range(1, len(cand) + 1)
        cand_top = cand.head(args.topn).copy()

        cand_path = os.path.join(outdir, f"candidates_diffusion_top{args.topn}_alpha{a:.2f}.csv")
        cand_top.to_csv(cand_path, index=False)

        qc = compute_qc(df, seed_nodes=seed_set, negative_nodes=neg_set, topk=args.qc_topk, llps_col=args.llps_col)
        qc_path = os.path.join(outdir, f"qc_alpha{a:.2f}.json")
        with open(qc_path, "w", encoding="utf-8") as f:
            json.dump(qc, f, indent=2, ensure_ascii=False)

        logger.info(f"[alpha={a:.2f}] Wrote full scores: {full_path}")
        logger.info(f"[alpha={a:.2f}] Wrote candidates:  {cand_path}")
        logger.info(
            f"[alpha={a:.2f}] QC seeds_in_top{args.qc_topk}={qc['seeds_in_topk']}/{qc['n_seeds']}, seed_rank_median={qc['seed_rank_median']}"
        )


if __name__ == "__main__":
    main()
