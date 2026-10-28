"""Loading and light harmonisation of the downloaded inputs. Parsed matrices are cached in data/interim."""

import gzip
import io
import re

import numpy as np
import pandas as pd

from .config import INTERIM, RAW


def _cache(name, build):
    p = INTERIM / f"{name}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    obj = build()
    obj.to_pickle(p)
    return obj


def gene_types() -> pd.DataFrame:
    """GENCODE v36 gene table (the GDC STAR annotation): versioned gene_id -> gene_name, gene_type."""
    g = pd.read_csv(RAW / "gdc" / "gencode_v36_gene_types.tsv", sep="\t", index_col=0)
    return g


def protein_coding_symbols() -> set:
    g = gene_types()
    return set(g.loc[g.gene_type == "protein_coding", "gene_name"])


def collapse_to_symbols(m: pd.DataFrame, id_to_symbol: pd.Series) -> pd.DataFrame:
    """Map Ensembl rows to symbols; for duplicated symbols keep the row with the highest mean (no row mixing)."""
    m = m[~m.index.str.contains("PAR_Y")]
    sym = id_to_symbol.reindex(m.index)
    keep = sym.notna().values
    m = m[keep]
    m.index = sym.values[keep]
    # sort by position (labels are not unique yet), highest mean first, then drop later duplicates
    order = np.argsort(-m.mean(axis=1).values, kind="stable")
    m = m.iloc[order]
    return m[~m.index.duplicated(keep="first")]


def tcga_matrix(kind: str = "star_tpm") -> pd.DataFrame:
    """TCGA LUAD+LUSC matrix from Xena, log2(x+1), symbols x samples (all samples, before cohort selection)."""

    def build():
        g = gene_types().gene_name
        parts = [pd.read_csv(RAW / "xena" / f"TCGA-{p}.{kind}.tsv.gz", sep="\t", index_col=0) for p in ("LUAD", "LUSC")]
        m = pd.concat(parts, axis=1)
        return collapse_to_symbols(m, g).astype("float32")

    return _cache(f"tcga_{kind}", build)


def tcga_library_size() -> pd.Series:
    """Total STAR reads assigned to genes per sample (from log2(count+1))."""

    def build():
        parts = [pd.read_csv(RAW / "xena" / f"TCGA-{p}.star_counts.tsv.gz", sep="\t", index_col=0) for p in ("LUAD", "LUSC")]
        m = pd.concat(parts, axis=1)
        return (np.power(2.0, m) - 1).sum(axis=0)

    return _cache("tcga_library_size", build)


def tcga_clinical() -> pd.DataFrame:
    parts = []
    for p in ("LUAD", "LUSC"):
        d = pd.read_csv(RAW / "xena" / f"TCGA-{p}.clinical.tsv.gz", sep="\t", low_memory=False)
        d["project"] = p
        parts.append(d)
    return pd.concat(parts).drop_duplicates("sample").set_index("sample")


def legacy_clinical() -> pd.DataFrame:
    """Firehose-legacy patient table (tumour location, smoking category) from cBioPortal."""
    return pd.concat([pd.read_csv(RAW / "cbioportal" / f"{s}.clinical_patient.tsv", sep="\t", index_col=0) for s in ("luad_tcga", "lusc_tcga")])


def pancan_quality() -> pd.DataFrame:
    q = pd.read_csv(RAW / "pancan" / "merged_sample_quality_annotations.tsv", sep="\t", low_memory=False)
    return q[q["cancer type"].isin(["LUAD", "LUSC"])]


def absolute_purity() -> pd.Series:
    p = pd.read_csv(RAW / "pancan" / "TCGA_mastercalls.abs_tables_JSedit.fixed.txt", sep="\t")
    p["s15"] = p["array"].str[:15]
    return p.drop_duplicates("s15").set_index("s15").purity


def gdc_annotations() -> pd.DataFrame:
    return pd.read_csv(RAW / "gdc" / "annotations.tsv", sep="\t")


def gdc_aliquots() -> pd.DataFrame:
    return pd.read_csv(RAW / "gdc" / "gdc_star_counts_aliquots.tsv", sep="\t")


def rna_integrity() -> pd.Series:
    """Mean RIN per sample (16-character barcode), from GDC analyte records."""
    a = pd.read_csv(RAW / "gdc" / "analytes.tsv", sep="\t")
    a = a[a.analyte_type.astype(str).str.contains("RNA")].copy()
    a["s16"] = a.analyte.astype(str).str[:16]
    a["rin"] = pd.to_numeric(a.rna_integrity_number, errors="coerce")
    return a.groupby("s16").rin.mean()


def slide_pathology() -> pd.DataFrame:
    """Pathologist slide review per sample: % tumour nuclei, % necrosis etc. (mean over top/bottom slides)."""
    s = pd.read_csv(RAW / "gdc" / "slides_pathology.tsv", sep="\t")
    s["s16"] = s.slide.astype(str).str[:16]
    cols = ["percent_tumor_nuclei", "percent_tumor_cells", "percent_normal_cells", "percent_stromal_cells", "percent_necrosis"]
    for c in cols:
        s[c] = pd.to_numeric(s[c], errors="coerce")
    return s.groupby("s16")[cols].mean()


def dna_panel() -> pd.DataFrame:
    """Binary DNA features (driver mutations, amplifications, CDKN2A deletion) per 15-character sample barcode."""
    out = []
    for st in ("luad_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018"):
        c = pd.read_csv(RAW / "cbioportal" / f"{st}_gistic_panel.tsv", sep="\t", index_col=0).T
        m = pd.read_csv(RAW / "cbioportal" / f"{st}_mutations_panel.tsv", sep="\t")
        m = m[~m.mutationType.isin(["Silent", "Intron", "3'UTR", "5'UTR", "RNA", "Splice_Region"])]
        f = pd.DataFrame(index=c.index)
        for g in ("SOX2", "TP63", "NKX2-1", "FGFR1", "EGFR", "MET", "ERBB2", "KRAS"):
            if g in c.columns:
                f["amp_" + g] = (c[g] >= 2).astype(int)
        f["del_CDKN2A"] = (c["CDKN2A"] <= -2).astype(int)
        mm = pd.crosstab(m.sampleId, m.gene).clip(upper=1).add_prefix("mut_")
        out.append(f.join(mm, how="left").fillna(0))
    d = pd.concat(out).fillna(0)
    d.index = d.index.str[:15]
    return d[~d.index.duplicated()]


# ---------------------------------------------------------------- external cohorts

def cptac3() -> tuple:
    """CPTAC-3 lung tumours (same GDC STAR pipeline as TCGA): log2(TPM+1) symbols x patients, and sample sheet."""

    def build():
        m = pd.read_csv(RAW / "external" / "cptac3" / "cptac3_tumour_tpm.tsv.gz", sep="\t", index_col=0)
        return np.log2(collapse_to_symbols(m, gene_types().gene_name) + 1).astype("float32")

    x = _cache("cptac3_tpm", build)
    sel = pd.read_csv(RAW / "external" / "cptac3" / "cptac3_selected_tumour_files.tsv", sep="\t").drop_duplicates("patient").set_index("patient")
    sel = sel.loc[x.columns]
    y = (sel.disease == "Squamous Cell Neoplasms").astype(int).values
    return x, y, sel


def _series_matrix(acc):
    txt = gzip.open(RAW / "external" / "geo" / f"{acc}_series_matrix.txt.gz", "rt", errors="ignore").read()
    head, body = txt.split("!series_matrix_table_begin\n")
    body = body.split("!series_matrix_table_end")[0]
    meta = {}
    for line in head.splitlines():
        if line.startswith(("!Sample_characteristics_ch1", "!Sample_geo_accession", "!Sample_title", "!Sample_source_name_ch1")):
            vals = [v.strip().strip('"') for v in line.split("\t")[1:]]
            key = line.split("\t")[0]
            if key == "!Sample_characteristics_ch1":
                key = pd.Series(vals).str.split(":").str[0].mode()[0]
                vals = list(pd.Series(vals).str.split(": ", n=1).str[1])
            meta.setdefault(key, vals)
    return body, pd.DataFrame(meta)


def _gpl570_symbols():
    ann = pd.read_csv(RAW / "external" / "geo" / "GPL570.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"], dtype=str).dropna()
    return ann[~ann["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]


def geo_array(acc) -> tuple:
    """Affymetrix U133 Plus 2 cohort: gene x sample (max probe), labels (1 = LUSC, 0 = LUAD) and sample metadata."""
    labels = {
        "GSE30219": ("histology", {"ADC": 0, "SQC": 1, "BAS": 1}),
        "GSE37745": ("histology", {"adeno": 0, "squamous": 1}),
        "GSE50081": ("histology", {"adenocarcinoma": 0, "squamous cell carcinoma": 1}),
        "GSE43580": ("!Sample_source_name_ch1", None),
    }

    def build():
        body, meta = _series_matrix(acc)
        x = pd.read_csv(io.StringIO(body), sep="\t", index_col=0)
        x.index = x.index.astype(str)
        sym = _gpl570_symbols()
        x = x.loc[x.index.intersection(sym.index)]
        x["sym"] = sym.reindex(x.index).values
        x = x.groupby("sym").max()
        if x.max().max() > 50:
            x = np.log2(x.clip(lower=1))
        return x.astype("float32")

    x = _cache(f"geo_{acc}", build)
    _, meta = _series_matrix(acc)
    meta = meta.set_index("!Sample_geo_accession").reindex(x.columns)
    col, mp = labels[acc]
    if acc == "GSE43580":
        lab = meta[col].str.split(",").str[0].map({"Adenocarcinoma": 0, "Squamous Cell Carcinoma": 1})
    else:
        lab = meta[col].map(mp)
    keep = lab.notna().values
    return x.loc[:, keep], lab[keep].astype(int).values, meta[keep]


def _fpkm_to_log_tpm(f):
    f = f.clip(lower=0)
    return np.log2(f / f.sum() * 1e6 + 1)


def uppsala_rnaseq() -> tuple:
    """GSE81089 (Uppsala RNA-seq; histology re-reviewed 2013): 1 = squamous, 2 = adenocarcinoma, 3 = large cell/NOS."""

    def build():
        f = pd.read_csv(RAW / "external" / "geo" / "GSE81089_FPKM_cufflinks.tsv.gz", sep="\t", index_col=0)
        g = gene_types()
        g = g[~g.index.str.contains("PAR_Y")]
        g.index = g.index.str.split(".").str[0]
        g = g[~g.index.duplicated()]
        f.index = f.index.str.split(".").str[0]
        f = f[f.index.isin(g.index)]
        return _fpkm_to_log_tpm(collapse_to_symbols(f, g.gene_name)).astype("float32")

    x = _cache("geo_GSE81089", build)
    _, meta = _series_matrix("GSE81089")
    meta["code"] = meta["histology"].astype(str).str.extract(r"(\d)")[0]
    cols = {c.split("_")[0]: c for c in x.columns}
    meta["col"] = meta["!Sample_title"].map(cols)
    t = meta[meta.code.isin(["1", "2"]) & meta.col.notna()]
    return x[t.col.values], (t.code == "1").astype(int).values, t


def korean_luad() -> tuple:
    """GSE40419 (Seoul; Seo 2012): 87 LUAD tumours and 77 adjacent normal lungs, log2 TPM (from RPKM)."""

    def build():
        k = pd.read_csv(RAW / "external" / "geo" / "GSE40419_LC-87_RPKM_expression.txt.gz", sep="\t").set_index("gene").iloc[:, 5:]
        k = k[~k.index.duplicated()]
        return _fpkm_to_log_tpm(k).astype("float32")

    x = _cache("geo_GSE40419", build)
    _, meta = _series_matrix("GSE40419")
    meta = meta.set_index("!Sample_title").reindex(x.columns)
    is_normal = np.array([c.endswith("_nor") for c in x.columns])
    return x, is_normal, meta


def barcode_parts(barcode: str) -> dict:
    """Split a TCGA barcode (see GDC 'TCGA Barcode' documentation)."""
    m = re.match(r"^(TCGA)-(\w{2})-(\w{4})-(\d{2})([A-Z])?", barcode)
    if not m:
        raise ValueError(f"not a TCGA barcode: {barcode}")
    return dict(project=m.group(1), tss=m.group(2), participant=m.group(3), patient=f"TCGA-{m.group(2)}-{m.group(3)}",
                sample_type=m.group(4), vial=m.group(5))
