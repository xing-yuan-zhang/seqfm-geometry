from __future__ import annotations

import argparse, hashlib, json, logging, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests, yaml

CHUNK = 1024 * 1024

UNIPROT_RE = re.compile(
    r"\b([A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9]|A0A[0-9A-Z]{7})\b"
)


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(CHUNK), b""):
            h.update(c)
    return h.hexdigest()


def _ensure(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def download(url: str, out: Path, overwrite=False, timeout=60):
    _ensure(out)
    if out.exists() and not overwrite:
        return

    tmp = out.with_suffix(out.suffix + ".partial")
    headers, offset = {}, (tmp.stat().st_size if tmp.exists() else 0)
    if offset:
        headers["Range"] = f"bytes={offset}-"

    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("ab" if offset else "wb") as f:
            for c in r.iter_content(chunk_size=CHUNK):
                if c:
                    f.write(c)

    tmp.replace(out)
    print(f"{out} ({out.stat().st_size/1e6:.2f} MB)")


def dataloader_main(
    root=".",
    string_version="v12.0",
    taxid="9606",
    with_biogrid=False,
    with_elm=False,
    overwrite=False,
):
    root = Path(root).resolve()
    base = "https://stringdb-downloads.org/download"
    string_dir = root / "inputs/ppi/STRING" / string_version

    files = [
        f"{taxid}.protein.links.detailed.{string_version}.txt.gz",
        f"{taxid}.protein.aliases.{string_version}.txt.gz",
        f"{taxid}.protein.info.{string_version}.txt.gz",
        f"{taxid}.protein.physical.links.detailed.{string_version}.txt.gz",
        f"{taxid}.protein.sequences.{string_version}.fa.gz",
    ]
    for fn in files:
        try:
            download(f"{base}/{fn}", string_dir / fn, overwrite=overwrite)
        except Exception as e:
            print(f"STRING failed: {fn}: {e}")

    if with_biogrid:
        bg_dir = root / "inputs/ppi/BioGRID"
        bg_base = "https://downloads.thebiogrid.org/BioGRID/Latest-Release"
        for fn in ["BIOGRID-ALL-LATEST.mitab.zip", "BIOGRID-MV-Physical-LATEST.mitab.zip"]:
            try:
                download(f"{bg_base}/{fn}", bg_dir / fn, overwrite=overwrite)
            except Exception as e:
                print(f"BioGRID failed: {fn}: {e}")

    if with_elm:
        elm_dir = root / "inputs/annotations/motifs/ELM"
        url = "http://elm.eu.org/instances.tsv?q=None&taxon=Homo%20sapiens&instance_logic=true%20positive"
        try:
            download(url, elm_dir / "ELM_instances_human_true_positive.tsv", overwrite=overwrite)
        except Exception as e:
            print(f"ELM failed: {e}")


@dataclass
class UniProtClientConfig:
    endpoint: str = "https://rest.uniprot.org"
    timeout_sec: int = 30
    max_retries: int = 3
    sleep_sec_between_calls: float = 0.1


class UniProtClient:
    def __init__(self, cfg: UniProtClientConfig, cache_path: Optional[str] = None):
        self.cfg = cfg
        self.cache_path = cache_path
        self.cache = {}
        if cache_path:
            try:
                self.cache = json.load(open(cache_path, "r", encoding="utf-8"))
            except FileNotFoundError:
                pass

    def _save(self):
        if not self.cache_path:
            return
        p = Path(self.cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.cache, p.open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    def search(self, query: str, taxon_id: int, fields=None):
        if fields is None:
            fields = [
                "accession",
                "id",
                "protein_name",
                "gene_primary",
                "gene_names",
                "organism_id",
                "length",
                "reviewed",
            ]
        key = f"{query}|{taxon_id}|{','.join(fields)}"
        if key in self.cache:
            return self.cache[key]

        url = f"{self.cfg.endpoint}/uniprotkb/search"
        params = {
            "query": f"({query}) AND (organism_id:{taxon_id})",
            "format": "json",
            "fields": ",".join(fields),
            "size": 5,
        }

        last = None
        for i in range(self.cfg.max_retries):
            try:
                r = requests.get(url, params=params, timeout=self.cfg.timeout_sec)
                if r.status_code == 400:
                    print("[UniProt 400]", r.text)
                r.raise_for_status()
                out = r.json()
                self.cache[key] = out
                self._save()
                time.sleep(self.cfg.sleep_sec_between_calls)
                return out
            except Exception as e:
                last = e
                time.sleep(0.5 * (i + 1))
        raise RuntimeError(f"UniProt query failed: {query}") from last


@dataclass
class Scope:
    taxon_id: int
    canonical_id: str
    allow_multiple_uniprot_hits: bool = False
    fail_on_unmapped: bool = True
    fail_on_ambiguous: bool = True


def setup_logger(path: str, level="INFO"):
    lg = logging.getLogger("seed_normalize")
    lg.setLevel(getattr(logging, level.upper(), logging.INFO))
    lg.handlers.clear()

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    lg.addHandler(fh)
    lg.addHandler(sh)
    return lg


def load_yaml(p: str):
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


def parse_scope(y) -> Scope:
    org = y.get("organism", {})
    pol = y.get("id_policy", {})
    qg = y.get("quality_gate", {})
    return Scope(
        taxon_id=int(org["taxon_id"]),
        canonical_id=pol["canonical_id"],
        allow_multiple_uniprot_hits=bool(pol.get("allow_multiple_uniprot_hits", False)),
        fail_on_unmapped=bool(qg.get("fail_on_unmapped", True)),
        fail_on_ambiguous=bool(qg.get("fail_on_ambiguous", True)),
    )


def best_hit(query: str, hits: list[dict]):
    q = query.upper()

    def gene_bits(h):
        g0 = (h.get("genes") or [{}])[0] or {}
        gp = ((g0.get("geneName") or {}).get("value") or "").upper()
        syn = {
            (s or {}).get("value", "").upper()
            for s in (g0.get("synonyms") or [])
            if isinstance((s or {}).get("value"), str)
        }
        return gp, syn

    def protein_name(h):
        return (
            ((h.get("proteinDescription") or {}).get("recommendedName") or {})
            .get("fullName", {})
            .get("value", "")
            .lower()
        )

    def score(h):
        reviewed = int("reviewed" in str(h.get("entryType", "")).lower())
        gp, syn = gene_bits(h)
        pname = protein_name(h)
        iso = int("isoform" in pname)
        length = int((h.get("sequence") or {}).get("length") or 0)
        return (reviewed, gp == q, q in syn, -iso, length)

    if not hits:
        return None
    ranked = sorted(hits, key=score, reverse=True)
    if len(ranked) == 1:
        return ranked[0]
    return None if score(ranked[0]) == score(ranked[1]) else ranked[0]


def hit_row(h):
    g0 = (h.get("genes") or [{}])[0] or {}
    gp = ((g0.get("geneName") or {}).get("value")) or ""
    pname = (
        ((h.get("proteinDescription") or {}).get("recommendedName") or {})
        .get("fullName", {})
        .get("value", "")
    )
    return {
        "uniprot_acc": h.get("primaryAccession", ""),
        "uniprot_id": h.get("uniProtkbId", ""),
        "gene_symbol": gp,
        "protein_name": pname,
        "taxon_id": (h.get("organism") or {}).get("taxonId", ""),
        "reviewed": h.get("entryType", ""),
        "length": (h.get("sequence") or {}).get("length", ""),
    }


def write_tsv(path: str, rows: list[dict]):
    if not rows:
        raise ValueError("no rows")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with p.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def uniprot_main(scope_path, seeds_path, cache_path, out_tsv, out_json, log_path):
    scope_y, seeds_y = load_yaml(scope_path), load_yaml(seeds_path)
    scope = parse_scope(scope_y)

    logger = setup_logger(log_path, scope_y.get("logging", {}).get("level", "INFO"))
    client = UniProtClient(UniProtClientConfig(**scope_y.get("uniprot", {})), cache_path)

    seeds = seeds_y.get("seeds", [])
    if not seeds:
        raise ValueError("seeds empty")

    rows, audit, unmapped, ambiguous = [], [], [], []

    for s in seeds:
        q = str(s["query"]).strip()
        raw = client.search(q, scope.taxon_id)
        hits = raw.get("results") or []

        if not hits:
            logger.error(f"UNMAPPED: {q}")
            unmapped.append(q)
            audit.append({"query": q, "status": "unmapped", "raw": raw})
            continue

        if len(hits) > 1 and not scope.allow_multiple_uniprot_hits:
            h = best_hit(q, hits)
            if h is None:
                logger.error(f"AMBIGUOUS: {q}")
                ambiguous.append(q)
                audit.append({"query": q, "status": "ambiguous", "raw": raw})
                continue
            hits = [h]

        row = hit_row(hits[0])
        row.update(
            query=q,
            role=s.get("role", ""),
            notes=s.get("notes", ""),
            evidence=json.dumps(s.get("evidence", []), ensure_ascii=False),
        )
        rows.append(row)
        audit.append({"query": q, "status": "ok", "resolved": row})
        logger.info(f"OK: {q} -> {row['uniprot_acc']}")

    if unmapped and scope.fail_on_unmapped:
        raise SystemExit(f"Fail unmapped: {unmapped}")
    if ambiguous and scope.fail_on_ambiguous:
        raise SystemExit(f"Fail ambiguous: {ambiguous}")

    write_tsv(out_tsv, rows)
    p = Path(out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "scope": {"taxon_id": scope.taxon_id, "canonical_id": scope.canonical_id},
            "seeds": rows,
            "audit": audit,
            "unmapped": unmapped,
            "ambiguous": ambiguous,
        },
        p.open("w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False,
    )
    logger.info(f"Wrote {out_tsv}")
    logger.info(f"Wrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("download")
    d.add_argument("--root", default=".")
    d.add_argument("--string-version", default="v12.0")
    d.add_argument("--taxid", default="9606")
    d.add_argument("--with-biogrid", action="store_true")
    d.add_argument("--with-elm", action="store_true")
    d.add_argument("--overwrite", action="store_true")

    u = sp.add_parser("uniprot")
    u.add_argument("--scope", required=True)
    u.add_argument("--seeds", required=True)
    u.add_argument("--cache", required=True)
    u.add_argument("--out_tsv", required=True)
    u.add_argument("--out_json", required=True)
    u.add_argument("--log", required=True)

    a = ap.parse_args()
    if a.cmd == "download":
        dataloader_main(
            root=a.root,
            string_version=a.string_version,
            taxid=a.taxid,
            with_biogrid=a.with_biogrid,
            with_elm=a.with_elm,
            overwrite=a.overwrite,
        )
    elif a.cmd == "uniprot":
        uniprot_main(a.scope, a.seeds, a.cache, a.out_tsv, a.out_json, a.log)
    else:
        raise SystemExit(a.cmd)


if __name__ == "__main__":
    main()
