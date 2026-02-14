import argparse
import gzip
import re
from pathlib import Path
import pandas as pd
import requests


def read_gz_tsv(path, sep="\t", **kwargs):
    kw = dict(sep=sep, dtype=str, low_memory=False, on_bad_lines="skip")
    kw.update(kwargs)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        return pd.read_csv(f, **kw)


def read_gz_space(path: Path):
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f, sep=" ")


def looks_like_uniprot(x: str) -> bool:
    return bool(UNIPROT_RE.fullmatch(str(x)))


def build_string_to_uniprot(aliases: pd.DataFrame) -> pd.DataFrame:
    if not {"string_id", "alias", "source"}.issubset(set(aliases.columns)):
        cols = list(aliases.columns)
        aliases = aliases.rename(columns={cols[0]: "string_id", cols[1]: "alias", cols[2]: "source"})

    ali = aliases.copy()
    ali["alias"] = ali["alias"].astype(str).str.strip()
    ali = ali[ali["alias"].map(looks_like_uniprot)].copy()

    def pri(src):
        s = str(src).lower()
        if "uniprot" in s:
            return 0
        if "ensembl" in s:
            return 1
        return 2

    ali["priority"] = ali["source"].map(pri)
    ali = ali.sort_values(["string_id", "priority"])
    best = ali.drop_duplicates("string_id", keep="first")[["string_id", "alias", "source"]]
    return best.rename(columns={"alias": "entry", "source": "source_hint"})


def extract_uniprot_from_biogrid_field(s):
    if s is None:
        return None
    s = str(s)

    for p in s.split("|"):
        p = p.strip()
        token = p.split(":", 1)[1].strip() if ":" in p else p
        if looks_like_uniprot(token):
            return token

    m = UNIPROT_RE.search(s)
    return m.group(1) if m else None


def preprocess_string(root: Path, string_version: str, taxid: str, outdir: Path):
    root = Path(root)
    string_dir = root / "inputs" / "ppi" / "STRING" / string_version
    detailed = string_dir / f"{taxid}.protein.links.detailed.{string_version}.txt.gz"
    aliases = string_dir / f"{taxid}.protein.aliases.{string_version}.txt.gz"
    info = string_dir / f"{taxid}.protein.info.{string_version}.txt.gz"

    outdir.mkdir(parents=True, exist_ok=True)

    df = read_gz_space(detailed)
    p1, p2 = df.columns[0], df.columns[1]
    if "combined_score" not in df.columns:
        raise ValueError("missing combined_score")

    keep = [p1, p2, "combined_score"]
    for c in ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database", "textmining"]:
        if c in df.columns:
            keep.append(c)

    edges = df[keep].copy()
    edges.columns = ["node_a", "node_b"] + [c for c in edges.columns[2:]]
    edges.to_csv(outdir / "string_edges.tsv", sep="\t", index=False)

    ali = read_gz_tsv(aliases, sep="\t")
    ali = ali.rename(columns={ali.columns[0]: "string_id", ali.columns[1]: "alias", ali.columns[2]: "source"})
    ali.to_csv(outdir / "string_aliases.tsv", sep="\t", index=False)

    inf = read_gz_tsv(info, sep="\t")
    inf = inf.rename(columns={inf.columns[0]: "string_id"})
    inf.to_csv(outdir / "string_info.tsv", sep="\t", index=False)

    print(f"STRING: edges={len(edges):,}, aliases={len(ali):,}, info={len(inf):,}")


def preprocess_biogrid(root: Path, outdir: Path):
    root = Path(root)
    biogrid_zip = root / "inputs" / "ppi" / "BioGRID" / "BIOGRID-ALL-LATEST.mitab.zip"
    if not biogrid_zip.exists():
        return

    import zipfile

    with zipfile.ZipFile(biogrid_zip, "r") as z:
        txts = [n for n in z.namelist() if n.endswith(".txt")]
        if not txts:
            raise ValueError("BioGRID zip has no .txt inside")
        with z.open(txts[0]) as f:
            df = pd.read_csv(f, sep="\t", header=0, low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]

    col_alt_a = "Alt IDs Interactor A" if "Alt IDs Interactor A" in df.columns else None
    col_alt_b = "Alt IDs Interactor B" if "Alt IDs Interactor B" in df.columns else None
    col_tax_a = "Taxid Interactor A" if "Taxid Interactor A" in df.columns else None
    col_tax_b = "Taxid Interactor B" if "Taxid Interactor B" in df.columns else None
    if not all([col_alt_a, col_alt_b, col_tax_a, col_tax_b]):
        raise ValueError("MITAB header columns depreciated")

    tax_a = df[col_tax_a].astype(str)
    tax_b = df[col_tax_b].astype(str)
    is_human = tax_a.str.contains("taxid:9606", na=False) & tax_b.str.contains("taxid:9606", na=False)

    def norm_prefix(s: str) -> str:
        return s.replace("uniprot/swiss-prot:", "uniprotkb:").replace("uniprot/trembl:", "uniprotkb:")

    a = df[col_alt_a].astype(str).map(norm_prefix)
    b = df[col_alt_b].astype(str).map(norm_prefix)

    outdir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"a": a[is_human], "b": b[is_human]}).dropna()
    out = out[out["a"].str.contains("uniprotkb:", na=False) & out["b"].str.contains("uniprotkb:", na=False)]
    out.to_csv(outdir / "biogrid_human_edges.tsv", sep="\t", index=False)
    print(f"BioGRID: edges={len(out):,}")


def _strip_isoform(acc: str) -> str:
    s = str(acc).strip()
    return s.split("-", 1)[0]


def load_sec2pri_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str)
    if not {"secondary", "primary"}.issubset(set(df.columns)):
        raise ValueError(f"sec2pri map err")
    df = df.dropna(subset=["secondary", "primary"])
    return dict(zip(df["secondary"], df["primary"]))


def download_uniprot_sec2pri(taxid: str, out_path: Path, force=False, reviewed_only=False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return

    base = "https://rest.uniprot.org/uniprotkb/search"
    q = f"(organism_id:{taxid})"
    if reviewed_only:
        q += " AND (reviewed:true)"

    fields_try = [
        "accession,sec_acc",
        "accession,secondary_accession",
        "accession,secondary_accessions",
        "accession,secondaryAccession",
    ]

    text = None
    used = None
    for fields in fields_try:
        r = requests.get(base, params={"query": q, "format": "tsv", "fields": fields, "size": 500}, timeout=60)
        if r.status_code == 200 and r.text and "Accession" in r.text.splitlines()[0]:
            text = r.text
            used = fields
            break

    if text is None:
        raise RuntimeError("fields mismatch")

    rows = []
    for line in text.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        pri = parts[0].strip()
        sec_field = parts[1].strip()
        if not pri or not sec_field:
            continue
        for sec in sec_field.split(";"):
            sec = sec.strip()
            if sec:
                rows.append((sec, pri))

    df = pd.DataFrame(rows, columns=["secondary", "primary"]).drop_duplicates()
    df.to_csv(out_path, sep="\t", index=False)
    print(f"UniProt sec2pri map: taxid={taxid} reviewed_only={reviewed_only} fields={used} rows={len(df):,} -> {out_path}")


def iter_sec2pri_from_dat_gz(dat_gz: Path):
    secs = []
    with gzip.open(dat_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ID   "):
                secs = []
            elif line.startswith("AC   "):
                acs = line[5:].strip()
                for tok in acs.split(";"):
                    acc = tok.strip()
                    if acc:
                        secs.append(acc)
            elif line.startswith("//"):
                if secs:
                    pri = secs[0]
                    for s in secs[1:]:
                        yield s, pri
                secs = []


def build_sec2pri_from_mirror(raw_dir: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = [raw_dir / "uniprot_sprot_human.dat.gz", raw_dir / "uniprot_trembl_human.dat.gz"]
    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    seen = set()
    with open(out_path, "w", encoding="utf-8") as w:
        w.write("secondary\tprimary\n")
        for p in files:
            for s, pri in iter_sec2pri_from_dat_gz(p):
                key = (s, pri)
                if key in seen:
                    continue
                seen.add(key)
                w.write(f"{s}\t{pri}\n")
    print(f"wrote {out_path} rows={len(seen):,}")


def normalize_uniprot_series(s: pd.Series, sec2pri: dict[str, str]) -> pd.Series:
    x = s.astype(str).map(_strip_isoform)
    return x.map(lambda acc: sec2pri.get(acc, acc))


def merge_edges(
    root: Path,
    string_version: str,
    taxid: str,
    sec2pri_path: Path,
    refresh_sec2pri: bool,
    reviewed_only: bool,
    sec2pri_source: str,
    uniprot_raw_dir: Path,
):
    root = Path(root)
    outdir = root / "inputs" / "ppi"
    outdir.mkdir(parents=True, exist_ok=True)

    string_dir = outdir / "STRING" / string_version
    biogrid_dir = outdir / "BioGRID"

    string_edges = string_dir / "string_edges.tsv"
    string_aliases = string_dir / "string_aliases.tsv"
    biogrid_edges = biogrid_dir / "biogrid_human_edges.tsv"

    out_merged = outdir / "edges.merged.uniprot.tsv"
    out_map = outdir / "string2uniprot.tsv"

    aliases = pd.read_csv(string_aliases, sep="\t")
    s2u = build_string_to_uniprot(aliases)
    s2u.to_csv(out_map, sep="\t", index=False)
    s2u_map = dict(zip(s2u["string_id"], s2u["entry"]))

    se = pd.read_csv(string_edges, sep="\t")
    if not {"node_a", "node_b"}.issubset(set(se.columns)):
        raise ValueError("missing node_a/node_b")

    se["entry_a"] = se["node_a"].map(s2u_map)
    se["entry_b"] = se["node_b"].map(s2u_map)
    se = se.dropna(subset=["entry_a", "entry_b"]).copy()
    se = se[se["entry_a"] != se["entry_b"]].copy()

    if "combined_score" not in se.columns:
        raise ValueError("missing combined_score")

    se["string_score"] = pd.to_numeric(se["combined_score"], errors="coerce")
    se = se.dropna(subset=["string_score"]).copy()
    se["weight"] = se["string_score"] / 1000.0
    se["is_string"] = 1
    se["is_biogrid"] = 0

    keep = ["entry_a", "entry_b", "weight", "string_score", "is_string", "is_biogrid"]
    se = se[keep]
    print("STRING mapped edges:", len(se))

    merged = se
    if biogrid_edges.exists():
        bg = pd.read_csv(biogrid_edges, sep="\t")
        if not {"a", "b"}.issubset(set(bg.columns)):
            bg = bg.rename(columns={bg.columns[0]: "a", bg.columns[1]: "b"})

        bg["entry_a"] = bg["a"].map(extract_uniprot_from_biogrid_field)
        bg["entry_b"] = bg["b"].map(extract_uniprot_from_biogrid_field)
        bg = bg.dropna(subset=["entry_a", "entry_b"]).copy()
        bg = bg[bg["entry_a"] != bg["entry_b"]].copy()

        bg["weight"] = 0.57
        bg["string_score"] = pd.NA
        bg["is_string"] = 0
        bg["is_biogrid"] = 1
        bg = bg[keep]

        merged = pd.concat([se, bg], ignore_index=True)
        print(f"BioGRID edges: {len(bg):,}")

    sec2pri_path.parent.mkdir(parents=True, exist_ok=True)
    if sec2pri_source == "rest":
        download_uniprot_sec2pri(taxid=taxid, out_path=sec2pri_path, force=refresh_sec2pri, reviewed_only=reviewed_only)
    elif sec2pri_source == "mirror":
        if refresh_sec2pri or (not sec2pri_path.exists()):
            build_sec2pri_from_mirror(uniprot_raw_dir, sec2pri_path)
    else:
        raise ValueError("must be rest or mirror")

    sec2pri = load_sec2pri_map(sec2pri_path)
    merged["entry_a"] = normalize_uniprot_series(merged["entry_a"], sec2pri)
    merged["entry_b"] = normalize_uniprot_series(merged["entry_b"], sec2pri)
    merged = merged.dropna(subset=["entry_a", "entry_b"]).copy()
    merged = merged[merged["entry_a"] != merged["entry_b"]].copy()

    merged["string_score_num"] = pd.to_numeric(merged["string_score"], errors="coerce")

    a = merged["entry_a"].astype(str)
    b = merged["entry_b"].astype(str)
    merged["u"] = a.where(a < b, b)
    merged["v"] = b.where(a < b, a)

    merged = merged.sort_values(["u", "v", "is_string", "string_score_num"], ascending=[True, True, False, False])
    out = merged.drop_duplicates(["u", "v"], keep="first").copy()

    out = out[["u", "v", "weight", "string_score", "is_string", "is_biogrid"]].rename(
        columns={"u": "entry_a", "v": "entry_b"}
    )
    out.to_csv(out_merged, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("ppi")
    p.add_argument("--root", default=".")
    p.add_argument("--string-version", default="v12.0")
    p.add_argument("--taxid", default="9606")
    p.add_argument("--preprocess-only", action="store_true")
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--refresh-sec2pri", action="store_true")
    p.add_argument("--uniprot-reviewed-only", action="store_true")
    p.add_argument("--sec2pri-source", choices=["rest", "mirror"], default="rest")
    p.add_argument("--uniprot-raw-dir", default=None)
    p.add_argument("--sec2pri-path", default=None)

    m = sp.add_parser("mirror")
    m.add_argument("--root", default=".")
    m.add_argument("--raw", default=None)
    m.add_argument("--out", default=None)

    a = ap.parse_args()

    if a.cmd == "mirror":
        root = Path(a.root).resolve()
        raw_dir = Path(a.raw).resolve() if a.raw else (root / "inputs" / "ppi" / "UniProt" / "raw")
        out_path = Path(a.out).resolve() if a.out else (root / "inputs" / "ppi" / "UniProt" / "uniprot_sec2pri_9606.tsv")
        build_sec2pri_from_mirror(raw_dir, out_path)
        return

    root = Path(a.root).resolve()
    outdir_string = root / "inputs" / "ppi" / "STRING" / a.string_version
    outdir_biogrid = root / "inputs" / "ppi" / "BioGRID"
    outdir_string.mkdir(parents=True, exist_ok=True)
    outdir_biogrid.mkdir(parents=True, exist_ok=True)

    sec2pri_path = Path(a.sec2pri_path).resolve() if a.sec2pri_path else (
        root / "inputs" / "ppi" / "UniProt" / f"uniprot_sec2pri_{a.taxid}.tsv"
    )
    uniprot_raw_dir = Path(a.uniprot_raw_dir).resolve() if a.uniprot_raw_dir else (
        root / "inputs" / "ppi" / "UniProt" / "raw"
    )

    if a.merge_only:
        merge_edges(
            root=root,
            string_version=a.string_version,
            taxid=a.taxid,
            sec2pri_path=sec2pri_path,
            refresh_sec2pri=a.refresh_sec2pri,
            reviewed_only=a.uniprot_reviewed_only,
            sec2pri_source=a.sec2pri_source,
            uniprot_raw_dir=uniprot_raw_dir,
        )
        return

    preprocess_string(root, a.string_version, a.taxid, outdir_string)
    preprocess_biogrid(root, outdir_biogrid)
    if not a.preprocess_only:
        merge_edges(
            root=root,
            string_version=a.string_version,
            taxid=a.taxid,
            sec2pri_path=sec2pri_path,
            refresh_sec2pri=a.refresh_sec2pri,
            reviewed_only=a.uniprot_reviewed_only,
            sec2pri_source=a.sec2pri_source,
            uniprot_raw_dir=uniprot_raw_dir,
        )


CHUNK = 1024 * 1024

UNIPROT_RE = re.compile(
    r"""
    \b(
        [A-NR-Z][0-9][A-Z0-9]{3}[0-9]
      | [OPQ][0-9][A-Z0-9]{3}[0-9]
      | A0A[0-9A-Z]{7}
    )\b
    """,
    re.VERBOSE,
)

if __name__ == "__main__":
    main()
