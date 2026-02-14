import argparse
import heapq
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


def build_keep_mask(df: pd.DataFrame, min_string_weight: float, biogrid_physical_only: bool) -> pd.Series:
    keep = pd.Series(False, index=df.index)

    if "is_string" in df.columns:
        keep |= (df["is_string"] == 1) & (df["weight"].astype(float) >= float(min_string_weight))

    if "is_biogrid" in df.columns:
        kb = (df["is_biogrid"] == 1)
        if biogrid_physical_only and ("biogrid_physical" in df.columns):
            kb &= (df["biogrid_physical"] == 1)
        keep |= kb

    return keep


def _push_topk(h, item, k: int):
    if k <= 0:
        h.append(item)
        return
    if len(h) < k:
        heapq.heappush(h, item)
        return
    if item[0] > h[0][0]:
        heapq.heapreplace(h, item)


def scan_collect_neighbors(
    edges_path: Path,
    active: set[str],
    min_string_weight: float,
    biogrid_physical_only: bool,
    topk_string_per_node: int,
    chunksize: int = 2_000_000,
) -> set[str]:
    neigh = set()
    heaps = defaultdict(list) if topk_string_per_node > 0 else None

    for chunk in pd.read_csv(edges_path, sep="\t", chunksize=chunksize):
        miss = {"entry_a", "entry_b"} - set(chunk.columns)
        if miss:
            raise ValueError(f"edges file missing columns: {sorted(miss)}")

        c = chunk[build_keep_mask(chunk, min_string_weight, biogrid_physical_only)]
        if c.empty:
            continue

        a = c["entry_a"].astype(str)
        b = c["entry_b"].astype(str)

        ma = a.isin(active)
        mb = b.isin(active)

        if "is_biogrid" in c.columns:
            cb = c[c["is_biogrid"] == 1]
            if not cb.empty:
                ab = cb["entry_a"].astype(str)
                bb = cb["entry_b"].astype(str)
                neigh.update(bb[ab.isin(active)].tolist())
                neigh.update(ab[bb.isin(active)].tolist())

        if "is_string" in c.columns:
            cs = c[c["is_string"] == 1]
            if not cs.empty:
                as_ = cs["entry_a"].astype(str)
                bs_ = cs["entry_b"].astype(str)
                ws_ = cs["weight"].astype(float)

                ms_a = as_.isin(active)
                ms_b = bs_.isin(active)

                if topk_string_per_node <= 0:
                    neigh.update(bs_[ms_a].tolist())
                    neigh.update(as_[ms_b].tolist())
                else:
                    for src, dst, w in zip(as_[ms_a], bs_[ms_a], ws_[ms_a]):
                        _push_topk(heaps[src], (float(w), str(dst)), topk_string_per_node)
                    for src, dst, w in zip(bs_[ms_b], as_[ms_b], ws_[ms_b]):
                        _push_topk(heaps[src], (float(w), str(dst)), topk_string_per_node)

    if topk_string_per_node > 0:
        for h in heaps.values():
            for _, dst in h:
                neigh.add(dst)

    return neigh


def scan_filter_edges(
    edges_path: Path,
    nodes: set[str],
    min_string_weight: float,
    biogrid_physical_only: bool,
    out_path: Path,
    chunksize: int = 2_000_000,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    first = True

    for chunk in pd.read_csv(edges_path, sep="\t", chunksize=chunksize):
        miss = {"entry_a", "entry_b"} - set(chunk.columns)
        if miss:
            raise ValueError(f"edges file missing columns: {sorted(miss)}")

        c = chunk[build_keep_mask(chunk, min_string_weight, biogrid_physical_only)]
        if c.empty:
            continue

        a = c["entry_a"].astype(str)
        b = c["entry_b"].astype(str)
        sub = c[a.isin(nodes) & b.isin(nodes)]
        if sub.empty:
            continue

        sub.to_csv(out_path, sep="\t", index=False, mode="w" if first else "a", header=first)
        first = False
        wrote += len(sub)

    return wrote


def pick_seed_set(seeds_df: pd.DataFrame, seed_roles: list[str], allow_missing_role: bool) -> set[str]:
    if "entry" not in seeds_df.columns:
        raise ValueError("need entry column")

    df = seeds_df.copy()
    df["entry"] = df["entry"].astype(str).str.strip()

    if "role" not in df.columns:
        if allow_missing_role:
            return set(df.dropna(subset=["entry"])["entry"].tolist())
        raise ValueError()

    df["role"] = df["role"].astype(str).str.strip()
    df = df[df["role"].isin(seed_roles)].dropna(subset=["entry"])
    return set(df["entry"].tolist())


def _sigmoid(x: float) -> float:
    if x >= 40:
        return 1.0
    if x <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def soft_rank_subgraph(
    edges_path: Path,
    base_nodes: set[str],
    seed_scores: dict[str, float],
    steps: int,
    decay: float,
    min_string_weight: float,
    temp: float,
    string_power: float,
    biogrid_physical_only: bool,
    biogrid_boost: float,
    frontier_topk: int,
    chunksize: int = 2_000_000,
) -> dict[str, float]:
    scores = defaultdict(float)
    frontier = dict(seed_scores)

    for k, v in seed_scores.items():
        if k in base_nodes:
            scores[k] = max(scores[k], float(v))

    for _ in range(int(steps)):
        if not frontier:
            break

        if frontier_topk > 0 and len(frontier) > frontier_topk:
            frontier = dict(sorted(frontier.items(), key=lambda x: x[1], reverse=True)[:frontier_topk])

        nxt = defaultdict(float)
        active = set(frontier.keys())

        for chunk in pd.read_csv(edges_path, sep="\t", chunksize=chunksize):
            miss = {"entry_a", "entry_b"} - set(chunk.columns)
            if miss:
                raise ValueError(f"edges file missing columns: {sorted(miss)}")

            c = chunk[build_keep_mask(chunk, min_string_weight, biogrid_physical_only)]
            if c.empty:
                continue

            a = c["entry_a"].astype(str)
            b = c["entry_b"].astype(str)

            in_base = a.isin(base_nodes) & b.isin(base_nodes)
            if not in_base.any():
                continue

            c = c[in_base]
            if c.empty:
                continue

            a = c["entry_a"].astype(str)
            b = c["entry_b"].astype(str)

            ma = a.isin(active)
            mb = b.isin(active)
            if not (ma.any() or mb.any()):
                continue

            if "is_biogrid" in c.columns:
                cb = c[c["is_biogrid"] == 1]
                if not cb.empty:
                    ab = cb["entry_a"].astype(str)
                    bb = cb["entry_b"].astype(str)
                    m1 = ab.isin(active)
                    m2 = bb.isin(active)

                    for u, v in zip(ab[m1], bb[m1]):
                        su = frontier.get(str(u), 0.0)
                        if su > 0:
                            nxt[str(v)] += su * float(decay) * float(biogrid_boost)

                    for u, v in zip(bb[m2], ab[m2]):
                        su = frontier.get(str(u), 0.0)
                        if su > 0:
                            nxt[str(v)] += su * float(decay) * float(biogrid_boost)

            if "is_string" in c.columns:
                cs = c[c["is_string"] == 1]
                if not cs.empty:
                    as_ = cs["entry_a"].astype(str)
                    bs_ = cs["entry_b"].astype(str)
                    ws_ = cs["weight"].astype(float)

                    ms_a = as_.isin(active)
                    ms_b = bs_.isin(active)

                    t = float(temp) if float(temp) > 1e-9 else 1e-9
                    p = float(string_power)

                    for u, v, w in zip(as_[ms_a], bs_[ms_a], ws_[ms_a]):
                        su = frontier.get(str(u), 0.0)
                        if su <= 0:
                            continue
                        gate = _sigmoid((float(w) - float(min_string_weight)) / t)
                        ew = (max(float(w), 0.0) ** p) * gate
                        if ew > 0:
                            nxt[str(v)] += su * float(decay) * ew

                    for u, v, w in zip(bs_[ms_b], as_[ms_b], ws_[ms_b]):
                        su = frontier.get(str(u), 0.0)
                        if su <= 0:
                            continue
                        gate = _sigmoid((float(w) - float(min_string_weight)) / t)
                        ew = (max(float(w), 0.0) ** p) * gate
                        if ew > 0:
                            nxt[str(v)] += su * float(decay) * ew

        for k, v in nxt.items():
            if k in base_nodes and v > scores[k]:
                scores[k] = float(v)

        frontier = dict(nxt)

    return dict(scores)


def main():
    ROOT = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=str(ROOT / "inputs/ppi/edges.merged.uniprot.tsv"))
    ap.add_argument("--seeds", default=str(ROOT / "inputs/seeds/seeds.mapped.tsv"))
    ap.add_argument("--hops", type=int, default=2, choices=[1, 2])

    ap.add_argument("--min-string-weight", type=float, default=0.90, help="combined_score/1000 threshold, 0.85~0.95")
    ap.add_argument("--biogrid-physical-only", action="store_true", default=False)
    ap.add_argument("--topk-string-per-node", type=int, default=100, help="hard expansion only (STRING topK)")
    ap.add_argument("--max-nodes", type=int, default=50000)
    ap.add_argument("--outdir", default=str(ROOT / "inputs/ppi/subgraph"))

    ap.add_argument("--seed-roles", default="Src-family-kinase,CIP4-TOCA-family,FBP17,TOCA-1")
    ap.add_argument("--allow-missing-role", action="store_true", default=False)

    ap.add_argument("--patch-seed", default=None)
    ap.add_argument("--patch-hops", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--patch-min-string-weight", type=float, default=None)
    ap.add_argument("--patch-topk-string-per-node", type=int, default=None)

    ap.add_argument("--soft", action="store_true", default=False)
    ap.add_argument("--soft-steps", type=int, default=6)
    ap.add_argument("--soft-decay", type=float, default=0.75)
    ap.add_argument("--soft-temp", type=float, default=0.02)
    ap.add_argument("--soft-string-power", type=float, default=1.0)
    ap.add_argument("--soft-biogrid-boost", type=float, default=1.0)
    ap.add_argument("--soft-frontier-topk", type=int, default=4000)
    ap.add_argument("--soft-topm", type=int, default=20000)

    args = ap.parse_args()

    edges_path = Path(args.edges)
    seeds_path = Path(args.seeds)
    outdir = Path(args.outdir)

    seeds_df = pd.read_csv(seeds_path, sep="\t")
    seed_roles = [x.strip() for x in str(args.seed_roles).split(",") if x.strip()]
    seed_set = pick_seed_set(seeds_df, seed_roles=seed_roles, allow_missing_role=args.allow_missing_role)
    seed_set = {x for x in seed_set if x and str(x).lower() != "nan"}
    if not seed_set:
        raise ValueError("empty seed_set")

    n1 = scan_collect_neighbors(
        edges_path=edges_path,
        active=seed_set,
        min_string_weight=args.min_string_weight,
        biogrid_physical_only=args.biogrid_physical_only,
        topk_string_per_node=args.topk_string_per_node,
    )
    base_nodes = set(seed_set) | set(n1)

    if args.hops == 2:
        n2 = scan_collect_neighbors(
            edges_path=edges_path,
            active=base_nodes,
            min_string_weight=args.min_string_weight,
            biogrid_physical_only=args.biogrid_physical_only,
            topk_string_per_node=args.topk_string_per_node,
        )
        base_nodes |= set(n2)

    patch_seed = str(args.patch_seed).strip()
    if patch_seed and patch_seed.lower() != "none":
        patch_minw = args.min_string_weight if args.patch_min_string_weight is None else float(args.patch_min_string_weight)
        patch_topk = args.topk_string_per_node if args.patch_topk_string_per_node is None else int(args.patch_topk_string_per_node)

        if patch_seed in seed_set:
            p1 = scan_collect_neighbors(
                edges_path=edges_path,
                active={patch_seed},
                min_string_weight=patch_minw,
                biogrid_physical_only=args.biogrid_physical_only,
                topk_string_per_node=patch_topk,
            )
            patch_nodes = {patch_seed} | set(p1)

            if args.patch_hops >= 2:
                p2 = scan_collect_neighbors(
                    edges_path=edges_path,
                    active=patch_nodes,
                    min_string_weight=patch_minw,
                    biogrid_physical_only=args.biogrid_physical_only,
                    topk_string_per_node=patch_topk,
                )
                patch_nodes |= set(p2)

            if args.patch_hops >= 3:
                p3 = scan_collect_neighbors(
                    edges_path=edges_path,
                    active=patch_nodes,
                    min_string_weight=patch_minw,
                    biogrid_physical_only=args.biogrid_physical_only,
                    topk_string_per_node=patch_topk,
                )
                patch_nodes |= set(p3)

            before = len(base_nodes)
            base_nodes |= patch_nodes
            after = len(base_nodes)
            print(
                f"patch_seed={patch_seed} patch_hops={args.patch_hops} "
                f"minw={patch_minw} topk={patch_topk} added_nodes={after-before:,}"
            )
        else:
            print(f"patch_seed={patch_seed} not in seed_set.")

    if len(base_nodes) > args.max_nodes:
        raise ValueError(f"base node number {len(base_nodes)} exceeds max_nodes={args.max_nodes} (tune hops/topk/minw)")

    nodes = set(base_nodes)

    if args.soft:
        seed_scores = {s: 1.0 for s in seed_set}
        scores = soft_rank_subgraph(
            edges_path=edges_path,
            base_nodes=base_nodes,
            seed_scores=seed_scores,
            steps=args.soft_steps,
            decay=args.soft_decay,
            min_string_weight=args.min_string_weight,
            temp=args.soft_temp,
            string_power=args.soft_string_power,
            biogrid_physical_only=args.biogrid_physical_only,
            biogrid_boost=args.soft_biogrid_boost,
            frontier_topk=args.soft_frontier_topk,
        )

        cand = [(v, k) for k, v in scores.items() if k not in seed_set]
        cand.sort(reverse=True)

        topm = int(args.soft_topm)
        if topm <= 0:
            topm = len(cand)

        pick = [k for _, k in cand[:topm]]
        nodes = set(seed_set) | set(pick)

        if len(nodes) > args.max_nodes:
            nodes = set(seed_set) | set([k for _, k in cand[: max(0, args.max_nodes - len(seed_set))]])

        print(
            f"soft=1 base_nodes={len(base_nodes):,} picked_nodes={len(nodes):,} "
            f"steps={args.soft_steps} decay={args.soft_decay} temp={args.soft_temp} "
            f"topm={args.soft_topm} frontier_topk={args.soft_frontier_topk}"
        )

    outdir.mkdir(parents=True, exist_ok=True)
    out_nodes = outdir / "subgraph_nodes.tsv"
    out_edges = outdir / "subgraph_edges.tsv"

    nodes_sorted = sorted(nodes)
    pd.DataFrame({"entry": nodes_sorted, "is_seed": [int(x in seed_set) for x in nodes_sorted]}).to_csv(
        out_nodes, sep="\t", index=False
    )

    n_edges = scan_filter_edges(
        edges_path=edges_path,
        nodes=nodes,
        min_string_weight=args.min_string_weight,
        biogrid_physical_only=args.biogrid_physical_only,
        out_path=out_edges,
    )

    print(f"wrote {out_nodes} nodes={len(nodes):,} seeds={len(seed_set):,}")
    print(f"wrote {out_edges} edges={n_edges:,}")
    print(
        f"hops={args.hops} min_string_weight={args.min_string_weight} "
        f"topk_string_per_node={args.topk_string_per_node} biogrid_physical_only={args.biogrid_physical_only}"
    )
    print(f"seed_roles={seed_roles}")


if __name__ == "__main__":
    main()
