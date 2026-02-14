from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx
import pandas as pd

try:
    import yaml
except ImportError:
    raise SystemExit()


def setup_logger(path: str) -> logging.Logger:
    lg = logging.getLogger("candidate_diffusion")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(sh)

    return lg


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
        return pickle.load(f)


def load_seeds_yaml(path: str) -> Tuple[List[SeedItem], List[ControlItem]]:
    obj = yaml.safe_load(open(path, "r", encoding="utf-8"))

    seeds = []
    for x in obj.get("seeds", []) or []:
        seeds.append(
            SeedItem(
                query=str(x.get("query", "")).strip(),
                weight=float(x.get("weight", 1.0)),
                role=str(x.get("role", "")).strip(),
                evidence=list(x.get("evidence", []) or []),
            )
        )

    ctrls = []
    for x in obj.get("negative_controls", []) or []:
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
    seed_weight_col: Optional[str],
    seed_weight_clip: Tuple[float, float],
):
    missing = []
    active = []
    weights: Dict[str, float] = {}

    lo, hi = seed_weight_clip

    def node_mult(n: str) -> float:
        if not seed_weight_col:
            return 1.0
        try:
            v = float(G.nodes[n].get(seed_weight_col))
        except Exception:
            return 1.0
        if v <= 0:
            return 1.0
        return max(lo, min(hi, v))

    for s in seeds:
        q = s.query
        if not q or s.role in exclude_roles or q not in G:
            missing.append(q)
            continue

        w0 = float(s.weight) if s.weight else 1.0
        if w0 <= 0:
            missing.append(q)
            continue

        w = w0 * node_mult(q)
        weights[q] = weights.get(q, 0.0) + w
        active.append(q)

    if not weights:
        raise ValueError("no valid seeds")

    tot = sum(weights.values())
    pers = {k: v / tot for k, v in weights.items()}

    logger.info(f"seeds used: {len(pers)}")

    if seed_weight_col:
        try:
            mults = {k: node_mult(k) for k in pers}
            show = sorted(mults.items(), key=lambda x: x[1], reverse=True)[:10]
            logger.info(f"seed_weight_col={seed_weight_col} clip={seed_weight_clip} top={show}")
        except Exception:
            pass

    if missing:
        logger.warning(
            f"missing seeds: {len(missing)} -> {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    return pers, sorted(set(active)), sorted(set(missing))


def run_ppr(
    G: nx.Graph,
    alpha: float,
    personalization: Dict[str, float],
    weight_attr: Optional[str],
    max_iter: int,
    tol: float,
    logger: logging.Logger,
):
    logger.info(
        f"PPR alpha={alpha:.3f} restart={1-alpha:.3f} weight={weight_attr} max_iter={max_iter} tol={tol}"
    )
    try:
        return nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            weight=weight_attr,
            max_iter=max_iter,
            tol=tol,
        )
    except nx.PowerIterationFailedConvergence as e:
        raise RuntimeError("did not converge") from e


def add_degree_features(G: nx.Graph, df: pd.DataFrame, weight_attr: Optional[str]):
    df["degree"] = df["node"].map(dict(G.degree())).fillna(0).astype(int)
    if weight_attr:
        try:
            wdeg = dict(G.degree(weight=weight_attr))
            df["weighted_degree"] = df["node"].map(wdeg).fillna(0.0)
        except Exception:
            pass
    return df


def maybe_merge_node_attrs(df: pd.DataFrame, path: Optional[str], logger):
    if not path or not os.path.exists(path):
        return df

    na = pd.read_csv(path, sep="\t")
    if "entry" in na.columns and "node" not in na.columns:
        na = na.rename(columns={"entry": "node"})
    if "node" not in na.columns:
        logger.warning(f"bad node attribute: {path}")
        return df

    logger.info(f"merged node attributes: {path}")
    return df.merge(na, on="node", how="left")


def compute_qc(
    df: pd.DataFrame,
    seed_nodes: Set[str],
    neg_nodes: Set[str],
    topk: int,
    llps_col: str,
):
    df = df.copy()
    df["rank"] = range(1, len(df) + 1)

    seed_df = df[df["node"].isin(seed_nodes)]
    neg_df = df[df["node"].isin(neg_nodes)]

    qc = {
        "n_nodes": len(df),
        "topk": topk,
        "n_seeds": len(seed_nodes),
        "n_neg_controls": len(neg_nodes),
        "seeds_in_topk": int((seed_df["rank"] <= topk).sum()) if len(seed_df) else 0,
        "seed_rank_mean": float(seed_df["rank"].mean()) if len(seed_df) else None,
        "seed_rank_median": float(seed_df["rank"].median()) if len(seed_df) else None,
        "neg_rank_mean": float(neg_df["rank"].mean()) if len(neg_df) else None,
        "neg_rank_median": float(neg_df["rank"].median()) if len(neg_df) else None,
        "top10_nodes": df.head(10)[["node", "score"]].to_dict("records"),
        "seed_ranks": seed_df[["node", "rank", "score"]].to_dict("records"),
        "neg_control_ranks": neg_df[["node", "rank", "score"]].to_dict("records"),
    }

    if llps_col in df.columns:
        base = df[llps_col].fillna(0).astype(int)
        top = df.head(topk)[llps_col].fillna(0).astype(int)
        qc.update(
            {
                "llps_col": llps_col,
                "llps_base_rate": float(base.mean()),
                "llps_in_topk": int(top.sum()),
                "llps_rate_in_topk": float(top.mean()),
            }
        )

    return qc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--alpha", nargs="+", type=float, default=[0.7])
    p.add_argument("--topn", type=int, default=100)
    p.add_argument("--weight_attr", default="weight")
    p.add_argument("--qc_topk", type=int, default=50)
    p.add_argument("--exclude_seed_roles", nargs="*", default=[])
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--node_attributes", default="")
    p.add_argument("--llps_col", default="is_LLPS_any")
    p.add_argument("--seed_weight_col", default="seed_weight")
    p.add_argument("--seed_weight_clip_min", type=float, default=0.1)
    p.add_argument("--seed_weight_clip_max", type=float, default=10.0)
    return p.parse_args()


def main():
    ROOT = Path(__file__).resolve().parents[1]
    args = parse_args()

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(str(outdir / "diffusion.log"))

    G = load_graph(str(ROOT / args.graph))
    logger.info(f"Graph loaded: |V|={G.number_of_nodes():,} |E|={G.number_of_edges():,}")

    seeds, ctrls = load_seeds_yaml(str(ROOT / args.seeds))
    exclude = set(args.exclude_seed_roles or [])

    pers, active_seeds, missing = build_personalization(
        G,
        seeds,
        exclude,
        logger,
        seed_weight_col=(args.seed_weight_col.strip() or None),
        seed_weight_clip=(args.seed_weight_clip_min, args.seed_weight_clip_max),
    )

    seed_set = set(active_seeds)
    neg_set = {c.query for c in ctrls if c.query in G}

    node_attr = args.node_attributes.strip()
    if not node_attr:
        guess = Path(args.graph).with_name("nodes.final.tsv")
        if guess.exists():
            node_attr = str(guess)

    cfg = {
        "graph": args.graph,
        "seeds_yaml": args.seeds,
        "alphas": args.alpha,
        "topn": args.topn,
        "weight_attr": args.weight_attr or None,
        "exclude_seed_roles": sorted(exclude),
        "active_seeds": active_seeds,
        "missing_or_excluded_seeds": missing,
        "negative_controls_in_graph": sorted(neg_set),
    }
    json.dump(cfg, open(outdir / "diffusion_config.json", "w"), indent=2)

    weight_attr = args.weight_attr.strip() or None

    for a in args.alpha:
        if not (0 < a < 1):
            raise ValueError(f"bad alpha={a}")

        scores = run_ppr(G, a, pers, weight_attr, args.max_iter, args.tol, logger)
        df = pd.DataFrame({"node": scores.keys(), "score": scores.values()})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        df = add_degree_features(G, df, weight_attr)
        df = maybe_merge_node_attrs(df, node_attr, logger)

        df.to_csv(outdir / f"diffusion_scores_alpha{a:.2f}.csv", index=False)

        cand = df[~df["node"].isin(seed_set)].copy()
        cand["rank_excluding_seeds"] = range(1, len(cand) + 1)
        cand.head(args.topn).to_csv(
            outdir / f"candidates_diffusion_top{args.topn}_alpha{a:.2f}.csv",
            index=False,
        )

        qc = compute_qc(df, seed_set, neg_set, args.qc_topk, args.llps_col)
        json.dump(qc, open(outdir / f"qc_alpha{a:.2f}.json", "w"), indent=2)

        logger.info(
            f"[alpha={a:.2f}] seeds_in_top{args.qc_topk}={qc['seeds_in_topk']}/{qc['n_seeds']} "
            f"seed_rank_median={qc['seed_rank_median']}"
        )


if __name__ == "__main__":
    main()
