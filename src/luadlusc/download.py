"""Download every input from its original public, open-access source.

All sources are open access (no login): UCSC Xena GDC hub, GDC API (TCGA + CPTAC-3), GDC PanCanAtlas
publication files, cBioPortal public API and NCBI GEO. Files already present are skipped. A SHA-256
manifest of what was used for the published results is in `data/SHA256SUMS.txt`.
"""

import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .config import DATA, RAW

XENA = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
GDC = "https://api.gdc.cancer.gov"
CBIO = "https://www.cbioportal.org/api"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"

XENA_FILES = [f"TCGA-{p}.{k}.tsv.gz" for p in ("LUAD", "LUSC") for k in ("star_tpm", "star_counts", "clinical")] + [
    "gencode.v36.annotation.gtf.gene.probemap"]
PANCAN_FILES = {  # GDC PanCanAtlas publication page file UUIDs
    "merged_sample_quality_annotations.tsv": "1a7d7be8-675d-4e60-a105-19d4121bdebf",
    "TCGA_mastercalls.abs_tables_JSedit.fixed.txt": "4f277128-f793-4354-a13d-30cc7fe9f6b5",
}
GEO_FILES = {
    "GSE30219": ["matrix/GSE30219_series_matrix.txt.gz"],
    "GSE37745": ["matrix/GSE37745_series_matrix.txt.gz"],
    "GSE50081": ["matrix/GSE50081_series_matrix.txt.gz"],
    "GSE43580": ["matrix/GSE43580_series_matrix.txt.gz"],
    "GSE81089": ["matrix/GSE81089_series_matrix.txt.gz", "suppl/GSE81089_FPKM_cufflinks.tsv.gz"],
    "GSE40419": ["matrix/GSE40419_series_matrix.txt.gz", "suppl/GSE40419_LC-87_RPKM_expression.txt.gz"],
}
GPL570_ANNOT = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
DNA_PANEL = ["EGFR", "KRAS", "STK11", "KEAP1", "NFE2L2", "PIK3CA", "TP53", "KMT2D", "NOTCH1", "PTEN", "CDKN2A", "SMARCA4",
             "RBM10", "BRAF", "MET", "ERBB2", "SOX2", "TP63", "NKX2-1", "FGFR1", "CUL3", "U2AF1", "ALK", "ROS1", "RET", "FAT1", "NF1"]


def _get(url, data=None, headers=None, tries=5, timeout=300):
    err = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001 - network errors are retried, then raised
            err = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"download failed: {url}: {err}")


def _post_json(url, body):
    return json.loads(_get(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Accept": "application/json"}))


def _save(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def fetch_xena():
    for f in XENA_FILES:
        p = RAW / "xena" / f
        if not p.exists():
            print("xena", f, flush=True)
            _save(p, _get(f"{XENA}/{f}"))


def fetch_pancan():
    for name, uuid in PANCAN_FILES.items():
        p = RAW / "pancan" / name
        if not p.exists():
            print("pancan", name, flush=True)
            _save(p, _get(f"{GDC}/data/{uuid}"))


def fetch_gdc():
    out = RAW / "gdc"
    proj = {"op": "in", "content": {"field": "project.project_id", "value": ["TCGA-LUAD", "TCGA-LUSC"]}}
    if not (out / "annotations.tsv").exists():
        rows, frm = [], 0
        while True:
            d = _post_json(f"{GDC}/annotations", {"filters": proj, "size": 2000, "from": frm,
                                                   "fields": "entity_submitter_id,entity_type,case_submitter_id,category,classification,notes,status,project.project_id"})
            hits = d["data"]["hits"]
            rows += [{**h, "project": (h.get("project") or {}).get("project_id")} for h in hits]
            frm += len(hits)
            if not hits or frm >= d["data"]["pagination"]["total"]:
                break
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "annotations.tsv", sep="\t", index=False)
    if not (out / "analytes.tsv").exists() or not (out / "slides_pathology.tsv").exists():
        hits = _post_json(f"{GDC}/cases", {"filters": proj, "expand": "samples.portions.slides,samples.portions.analytes", "size": 2000, "format": "JSON"})["data"]["hits"]
        slides, analytes = [], []
        slide_keys = ["percent_tumor_nuclei", "percent_tumor_cells", "percent_normal_cells", "percent_stromal_cells", "percent_necrosis",
                      "percent_lymphocyte_infiltration", "percent_monocyte_infiltration", "percent_neutrophil_infiltration"]
        for c in hits:
            cid, pj = c.get("submitter_id"), (c.get("project") or {}).get("project_id")
            for s in c.get("samples") or []:
                for p in s.get("portions") or []:
                    for sl in p.get("slides") or []:
                        slides.append(dict(case=cid, project=pj, sample=s.get("submitter_id"), sample_type=s.get("sample_type"),
                                           slide=sl.get("submitter_id"), section=sl.get("section_location"), **{k: sl.get(k) for k in slide_keys}))
                    for a in p.get("analytes") or []:
                        analytes.append(dict(case=cid, sample=s.get("submitter_id"), analyte=a.get("submitter_id"), analyte_type=a.get("analyte_type"),
                                             a260_a280_ratio=a.get("a260_a280_ratio"), concentration=a.get("concentration"),
                                             rna_integrity_number=a.get("rna_integrity_number")))
        pd.DataFrame(slides).to_csv(out / "slides_pathology.tsv", sep="\t", index=False)
        pd.DataFrame(analytes).to_csv(out / "analytes.tsv", sep="\t", index=False)
    if not (out / "gdc_star_counts_aliquots.tsv").exists():
        filt = {"op": "and", "content": [{"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-LUAD", "TCGA-LUSC"]}},
                                         {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
                                         {"op": "=", "content": {"field": "analysis.workflow_type", "value": "STAR - Counts"}}]}
        fields = "file_id,cases.project.project_id,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type,cases.samples.portions.analytes.aliquots.submitter_id"
        hits = _post_json(f"{GDC}/files", {"filters": filt, "fields": fields, "format": "JSON", "size": 5000})["data"]["hits"]
        rows = []
        for h in hits:
            c = h["cases"][0]
            s = c["samples"][0]
            al = s["portions"][0]["analytes"][0]["aliquots"][0]["submitter_id"]
            rows.append(dict(file_id=h["file_id"], project=c["project"]["project_id"], patient=c["submitter_id"], sample=s["submitter_id"],
                             sample_type=s["sample_type"], aliquot=al, plate=al[21:25], center=al[26:28]))
        pd.DataFrame(rows).to_csv(out / "gdc_star_counts_aliquots.tsv", sep="\t", index=False)
    if not (out / "gencode_v36_gene_types.tsv").exists():
        fid = pd.read_csv(out / "gdc_star_counts_aliquots.tsv", sep="\t").file_id.iloc[0]
        f = pd.read_csv(io.BytesIO(_get(f"{GDC}/data/{fid}")), sep="\t", comment="#")
        f = f[f.gene_id.str.startswith("ENSG")]
        f[["gene_id", "gene_name", "gene_type"]].to_csv(out / "gencode_v36_gene_types.tsv", sep="\t", index=False)


def _cbio_clinical(study, level):
    rows = json.loads(_get(f"{CBIO}/studies/{study}/clinical-data?clinicalDataType={level}&projection=SUMMARY&pageSize=10000000",
                           headers={"Accept": "application/json"}))
    key = "sampleId" if level == "SAMPLE" else "patientId"
    return pd.DataFrame(rows).pivot_table(index=key, columns="clinicalAttributeId", values="value", aggfunc="first")


def fetch_cbioportal():
    out = RAW / "cbioportal"
    for st in ("luad_tcga", "lusc_tcga"):
        p = out / f"{st}.clinical_patient.tsv"
        if not p.exists():
            out.mkdir(parents=True, exist_ok=True)
            _cbio_clinical(st, "PATIENT").to_csv(p, sep="\t")
    genes = _post_json(f"{CBIO}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", DNA_PANEL)
    ez = {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in genes}
    inv = {v: k for k, v in ez.items()}
    for st in ("luad_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018"):
        pm, pc = out / f"{st}_mutations_panel.tsv", out / f"{st}_gistic_panel.tsv"
        if not pm.exists():
            m = pd.DataFrame(_post_json(f"{CBIO}/molecular-profiles/{st}_mutations/mutations/fetch?projection=DETAILED",
                                        {"entrezGeneIds": list(ez.values()), "sampleListId": f"{st}_all"}))
            m["gene"] = m.gene.apply(lambda g: g["hugoGeneSymbol"])
            m[["sampleId", "gene", "proteinChange", "mutationType"]].to_csv(pm, sep="\t", index=False)
        if not pc.exists():
            c = pd.DataFrame(_post_json(f"{CBIO}/molecular-profiles/{st}_gistic/molecular-data/fetch?projection=SUMMARY",
                                        {"entrezGeneIds": list(ez.values()), "sampleListId": f"{st}_all"}))
            c["gene"] = c.entrezGeneId.map(inv)
            c.pivot_table(index="gene", columns="sampleId", values="value", aggfunc="mean").to_csv(pc, sep="\t")


def fetch_cptac3():
    out = RAW / "external" / "cptac3"
    mat = out / "cptac3_tumour_tpm.tsv.gz"
    if mat.exists():
        return
    filt = {"op": "and", "content": [{"op": "in", "content": {"field": "cases.project.project_id", "value": ["CPTAC-3"]}},
                                     {"op": "=", "content": {"field": "cases.primary_site", "value": "Bronchus and lung"}},
                                     {"op": "=", "content": {"field": "data_type", "value": "Gene Expression Quantification"}},
                                     {"op": "=", "content": {"field": "access", "value": "open"}}]}
    f = "file_id,cases.submitter_id,cases.disease_type,cases.samples.submitter_id,cases.samples.sample_type,cases.demographic.race,cases.demographic.gender"
    hits = _post_json(f"{GDC}/files", {"filters": filt, "fields": f, "size": 5000})["data"]["hits"]
    rows = []
    for h in hits:
        c = h["cases"][0]
        s = c["samples"][0]
        dm = c.get("demographic") or {}
        rows.append(dict(file_id=h["file_id"], patient=c["submitter_id"], disease=c.get("disease_type"), sample=s["submitter_id"],
                         sample_type=s["sample_type"], race=dm.get("race"), sex=dm.get("gender")))
    g = pd.DataFrame(rows)
    sel = g[g.sample_type == "Primary Tumor"].sort_values(["patient", "sample"]).drop_duplicates("patient")
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "cptac3_selected_tumour_files.tsv", sep="\t", index=False)

    def one(fid):
        d = pd.read_csv(io.BytesIO(_get(f"{GDC}/data/{fid}")), sep="\t", comment="#", usecols=["gene_id", "tpm_unstranded"])
        return fid, d[d.gene_id.str.startswith("ENSG")].set_index("gene_id").tpm_unstranded

    with ThreadPoolExecutor(8) as ex:
        cols = dict(ex.map(one, sel.file_id))
    m = pd.DataFrame(cols)
    m.columns = sel.set_index("file_id").loc[m.columns, "patient"].values
    m.to_csv(mat, sep="\t", compression="gzip")


def fetch_geo():
    out = RAW / "external" / "geo"
    for acc, files in GEO_FILES.items():
        for rel in files:
            p = out / rel.split("/")[-1]
            if not p.exists():
                print("geo", p.name, flush=True)
                _save(p, _get(f"{GEO}/{acc[:5]}nnn/{acc}/{rel}"))
    p = out / "GPL570.annot.gz"
    if not p.exists():
        _save(p, _get(GPL570_ANNOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksums():
    """Compare local raw files with the manifest used for the published results."""
    man = DATA / "SHA256SUMS.txt"
    if not man.exists():
        print("no manifest found")
        return True
    ok = True
    for line in man.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        p = DATA / rel
        if not p.exists():
            print("MISSING", rel)
            ok = False
        elif sha256(p) != digest:
            print("CHANGED (source updated since publication?)", rel)
            ok = False
    print("checksums OK" if ok else "checksum differences found (results may differ slightly)")
    return ok


def write_checksums():
    files = sorted(p for p in RAW.rglob("*") if p.is_file() and not p.name.endswith(".part"))
    lines = [f"{sha256(p)}  {p.relative_to(DATA).as_posix()}" for p in files]
    (DATA / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote manifest for {len(files)} files")


def fetch_all():
    fetch_xena()
    fetch_pancan()
    fetch_gdc()
    fetch_cbioportal()
    fetch_geo()
    fetch_cptac3()  # slowest (337 files from GDC); last so the core data is usable first
    print("all inputs present")


if __name__ == "__main__":
    fetch_all()
