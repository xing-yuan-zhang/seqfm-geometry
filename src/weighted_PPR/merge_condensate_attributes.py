from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional, Set, Dict

import pandas as pd


UNIPROT_RE = re.compile(
    r"""(?x)
    \b(
        [OPQ][0-9][A-Z0-9]{3}[0-9]
      | [A-NR-Z][0-9][A-Z0-9]{3}[0-9]
      | A0A[0-9A-Z]{7}
    )\b
    """
)

def normalize_entry(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()

    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 3 and UNIPROT_RE.fullmatch(parts[1].strip()):
            s = parts[1].strip()

    if "-" in s:
        head = s.split("-", 1)[0].strip()
        if UNIPROT_RE.fullmatch(head):
            s = head

    m = UNIPROT_RE.search(s)
    if m:
        return m.group(1)

    return s


def split_semicolon(s: Any) -> list[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    toks = [t.strip() for t in str(s).split(";") if t.strip()]
    return toks


def union_semicolon(series: pd.Series) -> str:
    items: list[str] = []
    for x in series.dropna():
        items.extend(split_semicolon(x))
    seen = set()
    out = []
    for t in items:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return ";".join(out) + (";" if out else "")


def first_nonempty(series: pd.Series) -> str:
    for x in series.fillna("").astype(str):
        if x.strip():
            return x.strip()
    return ""


def _read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".xls", ".xlsx"]:
        try:
            return pd.read_excel(path, dtype=str)
        except Exception:
            return pd.read_csv(path, sep="\t", dtype=str, engine="python", on_bad_lines="skip")
    return pd.read_csv(path, sep="\t", dtype=str, engine="python", on_bad_lines="skip")


def _infer_col(df: pd.DataFrame, keywords: list[str]) -> Optional[str]:
    cols = list(df.columns)
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in keywords):
            return c
    return None


def parse_llpsdb(llpsdb_in: Path, llpsdb_out: Path) -> Path:
    llpsdb_out.parent.mkdir(parents=True, exist_ok=True)

    df = _read_any_table(llpsdb_in)
    if df.empty:
        raise ValueError(f"LLPSDB input is empty: {llpsdb_in}")

    uniprot_col = _infer_col(df, ["uniprot", "accession", "acc", "entry"])
    if uniprot_col is None:
        best = None
        best_rate = 0.0
        for c in df.columns:
            s = df[c].astype(str).map(lambda x: bool(UNIPROT_RE.search(str(x))))
            rate = float(s.mean())
            if rate > best_rate:
                best_rate = rate
                best = c
        if best is None or best_rate < 0.05:
            raise ValueError(f"no UniProt column in LLPSDB file: {llpsdb_in}")
        uniprot_col = best

    out = pd.DataFrame({
        "entry": df[uniprot_col].apply(normalize_entry),
        "is_LLPSDB": 1
    })
    out = out[out["entry"].ne("")].drop_duplicates(subset=["entry"])
    out.to_csv(llpsdb_out, sep="\t", index=False)
    print(f"LLPSDB positives: {len(out):,} -> {llpsdb_out.resolve()}")
    return llpsdb_out


def parse_phasepdb(phasepdb_in: Path, phasepdb_out: Path) -> Path:
    phasepdb_out.parent.mkdir(parents=True, exist_ok=True)

    df = _read_any_table(phasepdb_in)
    if df.empty:
        raise ValueError(f"PhaSepDB input is empty: {phasepdb_in}")

    uniprot_col = _infer_col(df, ["uniprot", "accession", "acc", "entry"])
    if uniprot_col is None:
        best = None
        best_rate = 0.0
        for c in df.columns:
            s = df[c].astype(str).map(lambda x: bool(UNIPROT_RE.search(str(x))))
            rate = float(s.mean())
            if rate > best_rate:
                best_rate = rate
                best = c
        if best is None or best_rate < 0.05:
            raise ValueError(f"no UniProt column in PhaSepDB file: {phasepdb_in}")
        uniprot_col = best

    out = pd.DataFrame({
        "entry": df[uniprot_col].apply(normalize_entry),
        "is_PhaSepDB": 1
    })
    out = out[out["entry"].ne("")].drop_duplicates(subset=["entry"])
    out.to_csv(phasepdb_out, sep="\t", index=False)
    print(f"PhaSepDB positives: {len(out):,} -> {phasepdb_out.resolve()}")
    return phasepdb_out


def _read_elm(elm_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        elm_path, sep="\t", comment="#", dtype=str,
        engine="python", on_bad_lines="skip"
    )


def _infer_uniprot_col_elm(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = c.lower()
        if cl in ["primary_acc", "primary accession", "primary_accession", "uniprot", "uniprot_id", "entry", "acc"]:
            return c
    best = None
    best_rate = 0.0
    for c in df.columns:
        rate = float(df[c].astype(str).map(lambda x: bool(UNIPROT_RE.search(str(x)))).mean())
        if rate > best_rate:
            best_rate = rate
            best = c
    if best is None or best_rate < 0.05:
        raise ValueError("no UniProt column in ELM data.")
    return best


def _infer_elm_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if ("elm" in cl and "id" in cl) or cl in ["elmid", "elmidentifier"]:
            return c
    return None


def build_subgraph_node_attributes(
    nodes_file: Path,
    interpro_file: Path,
    llpsdb_pos: Path,
    phasepdb_pos: Path,
    elm_file: Optional[Path],
    out_file: Path,
    sh3_pfams: Set[str],
) -> Path:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    nodes = pd.read_csv(nodes_file, sep="\t", dtype=str)
    if "entry" not in nodes.columns:
        raise ValueError(f"{nodes_file} must have column 'entry'")
    nodes["entry"] = nodes["entry"].apply(normalize_entry)

    if "is_seed" in nodes.columns:
        nodes["is_seed"] = pd.to_numeric(nodes["is_seed"], errors="coerce").fillna(0).astype(int)
    else:
        nodes["is_seed"] = 0

    ip = None
    if interpro_file.exists():
        ip0 = pd.read_csv(interpro_file, sep="\t", dtype=str, engine="python", on_bad_lines="skip")
        ip0 = ip0.rename(columns={
            "Entry": "entry",
            "Gene Names (synonym)": "gene_symbol",
            "InterPro": "interpro_list",
            "Pfam": "pfam_list",
            "Length": "length",
            "Subcellular location [CC]": "subcellular_location",
            "Entry Name": "entry_name",
        })

        for col in ["entry", "gene_symbol", "interpro_list", "pfam_list", "length", "subcellular_location", "entry_name"]:
            if col not in ip0.columns:
                ip0[col] = ""

        ip0["entry"] = ip0["entry"].apply(normalize_entry)
        ip0["gene_symbol"] = ip0["gene_symbol"].fillna("").astype(str).str.split().str[0]
        ip0["interpro_list"] = ip0["interpro_list"].fillna("")
        ip0["pfam_list"] = ip0["pfam_list"].fillna("")

        def has_any_pfam(s: str, target: Set[str]) -> bool:
            toks = [t.strip() for t in str(s).split(";") if t.strip()]
            return any(t in target for t in toks)

        ip0["has_SH3"] = ip0["pfam_list"].apply(lambda s: has_any_pfam(s, sh3_pfams))

        ip0["has_PRD"] = False

        ip0["interpro_n"] = ip0["interpro_list"].apply(lambda s: len(split_semicolon(s)))
        ip0["pfam_n"] = ip0["pfam_list"].apply(lambda s: len(split_semicolon(s)))
        ip0["has_any_domain"] = (ip0["interpro_n"] + ip0["pfam_n"] > 0)

        ip = (
            ip0.groupby("entry", as_index=False)
               .agg({
                   "gene_symbol": first_nonempty,
                   "entry_name": first_nonempty,
                   "interpro_list": union_semicolon,
                   "pfam_list": union_semicolon,
                   "length": "max",
                   "subcellular_location": first_nonempty,
                   "has_SH3": "max",
                   "has_PRD": "max",
                   "interpro_n": "max",
                   "pfam_n": "max",
                   "has_any_domain": "max",
               })
        )
    else:
        ip = pd.DataFrame({"entry": nodes["entry"].unique()})

    out = nodes.merge(ip, on="entry", how="left")

    out["is_LLPSDB"] = 0

    if llpsdb_pos.exists():
        llps = pd.read_csv(llpsdb_pos, sep="\t", dtype=str)
        if "entry" not in llps.columns:
            raise ValueError(f"{llpsdb_pos} must contain column 'entry'")
        llps["entry"] = llps["entry"].apply(normalize_entry)
        if "is_LLPSDB" not in llps.columns:
            llps["is_LLPSDB"] = 1
        llps["is_LLPSDB"] = pd.to_numeric(llps["is_LLPSDB"], errors="coerce").fillna(1).astype(int)
        llps = llps[["entry", "is_LLPSDB"]].drop_duplicates()
        out["is_LLPSDB"] = 0
        if llpsdb_pos.exists():
            llps = pd.read_csv(llpsdb_pos, sep="\t", dtype=str)
            if "entry" not in llps.columns:
                raise ValueError(f"{llpsdb_pos} must contain column 'entry'")
            llps["entry"] = llps["entry"].apply(normalize_entry)
            if "is_LLPSDB" not in llps.columns:
                llps["is_LLPSDB"] = 1
            llps["is_LLPSDB"] = pd.to_numeric(llps["is_LLPSDB"], errors="coerce").fillna(1).astype(int)
            llps = llps[["entry", "is_LLPSDB"]].drop_duplicates()

            out = out.merge(llps, on="entry", how="left", suffixes=("", "_llps"))

            out["is_LLPSDB"] = pd.to_numeric(out["is_LLPSDB_llps"], errors="coerce").fillna(out["is_LLPSDB"]).fillna(0).astype(int)
            out = out.drop(columns=["is_LLPSDB_llps"])

    out["is_PhaSepDB"] = 0

    if phasepdb_pos.exists():
        phase = pd.read_csv(phasepdb_pos, sep="\t", dtype=str)
        if "entry" not in phase.columns:
            raise ValueError(f"{phasepdb_pos} must contain column 'entry'")
        phase["entry"] = phase["entry"].apply(normalize_entry)
        if "is_PhaSepDB" not in phase.columns:
            phase["is_PhaSepDB"] = 1
        phase["is_PhaSepDB"] = pd.to_numeric(phase["is_PhaSepDB"], errors="coerce").fillna(1).astype(int)
        phase = phase[["entry", "is_PhaSepDB"]].drop_duplicates()
        out["is_PhaSepDB"] = 0
        if phasepdb_pos.exists():
            phase = pd.read_csv(phasepdb_pos, sep="\t", dtype=str)
            if "entry" not in phase.columns:
                raise ValueError(f"{phasepdb_pos} must contain column 'entry'")
            phase["entry"] = phase["entry"].apply(normalize_entry)
            if "is_PhaSepDB" not in phase.columns:
                phase["is_PhaSepDB"] = 1
            phase["is_PhaSepDB"] = pd.to_numeric(phase["is_PhaSepDB"], errors="coerce").fillna(1).astype(int)
            phase = phase[["entry", "is_PhaSepDB"]].drop_duplicates()

            out = out.merge(phase, on="entry", how="left", suffixes=("", "_phase"))
            out["is_PhaSepDB"] = pd.to_numeric(out["is_PhaSepDB_phase"], errors="coerce").fillna(out["is_PhaSepDB"]).fillna(0).astype(int)
            out = out.drop(columns=["is_PhaSepDB_phase"])

    out["is_LLPS_any"] = ((out["is_LLPSDB"] + out["is_PhaSepDB"]) > 0).astype(int)

    out["elm_total"] = 0
    out["elm_sh3_related"] = 0

    if elm_file is not None and Path(elm_file).exists():
        try:
            elm = pd.read_csv(elm_file, sep="\t", comment="#", dtype=str, engine="python", on_bad_lines="skip")

            if "Primary_Acc" in elm.columns:
                ucol = "Primary_Acc"
            else:
                best, best_rate = None, 0.0
                for c in elm.columns:
                    rate = float(elm[c].astype(str).map(lambda x: bool(UNIPROT_RE.search(str(x)))).mean())
                    if rate > best_rate:
                        best, best_rate = c, rate
                if best is None or best_rate < 0.05:
                    raise ValueError("no UniProt column in ELM data.")
                ucol = best

            elm[ucol] = elm[ucol].apply(normalize_entry)

            total = (
                elm.groupby(ucol)
                .size()
                .rename("elm_total_new")
                .reset_index()
                .rename(columns={ucol: "entry"})
            )

            id_col = None
            for c in elm.columns:
                cl = c.lower()
                if ("elm" in cl and "id" in cl) or cl in ["elmid", "elmidentifier"]:
                    id_col = c
                    break

            if id_col is not None:
                sh3 = elm[elm[id_col].astype(str).str.contains("SH3", case=False, na=False)]
                sh3_cnt = (
                    sh3.groupby(ucol)
                    .size()
                    .rename("elm_sh3_related_new")
                    .reset_index()
                    .rename(columns={ucol: "entry"})
                )
            else:
                sh3_cnt = None

            out = out.merge(total, on="entry", how="left")
            if sh3_cnt is not None:
                out = out.merge(sh3_cnt, on="entry", how="left")

            out["elm_total"] = (
                pd.to_numeric(out.get("elm_total_new", pd.Series(0, index=out.index)),
                            errors="coerce")
                .fillna(0)
                .astype(int)
            )

            out["elm_sh3_related"] = (
                pd.to_numeric(out.get("elm_sh3_related_new", pd.Series(0, index=out.index)),
                            errors="coerce")
                .fillna(0)
                .astype(int)
            )

            for c in ["elm_total_new", "elm_sh3_related_new"]:
                if c in out.columns:
                    out = out.drop(columns=[c])

        except Exception as e:
            print(f"ELM merge failed: {e}.")

    out["elm_total"] = pd.to_numeric(out["elm_total"], errors="coerce").fillna(0).astype(int)
    out["elm_sh3_related"] = pd.to_numeric(out["elm_sh3_related"], errors="coerce").fillna(0).astype(int)

    out["has_PRD"] = (out["elm_sh3_related"] > 0)

    for bcol in ["has_SH3", "has_PRD", "has_any_domain"]:
        if bcol in out.columns:
            out[bcol] = out[bcol].fillna(False).astype(bool)

    if "length" in out.columns:
        out["length"] = pd.to_numeric(out["length"], errors="coerce")

    n_total = len(out)
    n_sh3 = int(out["has_SH3"].sum()) if "has_SH3" in out.columns else 0
    n_llps = int(out["is_LLPS_any"].sum())
    print(f"subgraph nodes={n_total:,} | has_SH3={n_sh3:,} | LLPS_any={n_llps:,}")

    out.to_csv(out_file, sep="\t", index=False)
    print(f"wrote node attributes: {out_file.resolve()}")
    return out_file


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["all", "parse", "merge"], default="all")

    ap.add_argument("--nodes", required=True, help="subgraph_nodes.tsv")
    ap.add_argument("--interpro", required=True, help="interproscan_summary.tsv")

    ap.add_argument("--llpsdb_in", required=True, help="LLPSDB raw input")
    ap.add_argument("--phasepdb_in", required=True, help="PhaSepDB raw input")

    ap.add_argument("--llpsdb_out", required=True, help="LLPSDB positives out")
    ap.add_argument("--phasepdb_out", required=True, help="PhaSepDB positives out")

    ap.add_argument("--elm", default="", help="ELM instances file")

    ap.add_argument("--out", required=True, help="subgraph_node_attributes.tsv")
    ap.add_argument("--sh3_pfams", default="PF00018", help="Pfam IDs to call SH3 domains")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    
    nodes_file = Path(str(ROOT / args.nodes))
    interpro_file = Path(str(ROOT / args.interpro))
    llpsdb_in = Path(str(ROOT / args.llpsdb_in))
    phasepdb_in = Path(str(ROOT / args.phasepdb_in))
    llpsdb_out = Path(str(ROOT / args.llpsdb_out))
    phasepdb_out = Path(str(ROOT / args.phasepdb_out))
    out_file = Path(str(ROOT / args.out))
    elm_file = Path(str(ROOT / args.elm)) if args.elm.strip() else None

    sh3_pfams = {x.strip() for x in args.sh3_pfams.split(",") if x.strip()}

    if args.run in ["all", "parse"]:
        parse_llpsdb(llpsdb_in, llpsdb_out)
        parse_phasepdb(phasepdb_in, phasepdb_out)

    if args.run in ["all", "merge"]:
        build_subgraph_node_attributes(
            nodes_file=nodes_file,
            interpro_file=interpro_file,
            llpsdb_pos=llpsdb_out,
            phasepdb_pos=phasepdb_out,
            elm_file=elm_file,
            out_file=out_file,
            sh3_pfams=sh3_pfams,
        )


if __name__ == "__main__":
    cli()
