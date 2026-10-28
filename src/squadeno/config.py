"""Paths, seeds and fixed analysis constants. Everything that is a decision lives here, in one place."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("SQUADENO_DATA", ROOT / "data"))
RAW = DATA / "raw"
INTERIM = DATA / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"

for _d in (INTERIM, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Five seeds, as used across the analysis (fold assignment, subsampling, bootstraps).
SEEDS = (42, 123, 456, 789, 1337)
# Date printed on the write up (fixed, so rebuilding the report on another day gives the same file).
WRITEUP_DATE = "28 October 2026"
PRIMARY_SEED = 42

# Hyperparameter grid for the L2 logistic model (inverse regularisation strength).
C_GRID = (0.0001, 0.001, 0.01, 0.1)
N_OUTER, N_INNER = 5, 4

# A gene is "expressed" if log2(TPM+1) > 1 in more than 20% of training tumours (fit inside each fold).
EXPR_THRESHOLD, EXPR_MIN_FRACTION = 1.0, 0.20

# TCGA sample-type codes (barcode characters 14-15).
PRIMARY_TUMOUR, SOLID_NORMAL = "01", "11"

# Tissue-source-site names that refer to the same institution in the two projects.
# Merging makes institution groups larger, which is the conservative direction for a grouped split.
INSTITUTION_ALIASES = {
    "Ontario Institute for Cancer Research (OICR)": "Ontario Institute for Cancer Research",
    "Fox Chase Cancer Center": "Fox Chase",
    "St Joseph's Medical Center (MD)": "St. Joseph's Medical Center (MD)",
    "Washington University - Mayo Clinic": "Mayo Clinic - Rochester",
}

# GDC annotation categories that remove a case (treatment before resection, wrong specimen, off-spec results).
EXCLUDING_ANNOTATIONS = (
    "Neoadjuvant therapy",
    "Case submitted is found to be a recurrence after submission",
    "Molecular analysis outside specification",
)

# Corneocyte-lipid-envelope genes, taken verbatim from Akiyama 2017 (PMID 28623042), so the
# lipid module is pre-specified from the literature rather than chosen from our data.
AKIYAMA_CLE_GENES = ("ABCA12", "ALOXE3", "ALOX12B", "CYP4F22", "CERS3", "ABHD5", "PNPLA1", "ELOVL4")
# 2-hydroxy-hexosylceramide route reported in LUAD by Lemay 2019 (PMID 31409741).
LEMAY_GENES = ("FA2H", "UGT8")

# Published baselines.
# Li 2019 (PMID 31752667): LUSC if KRT5 > AGR2 within the sample.
LI2019_RULE = ("KRT5", "AGR2")
# Hamaneh 2022 (PMID 35722224), Table 2 weights; negative = higher in LUSC.
HAMANEH2022_WEIGHTS = {"TP63": -0.662, "CALML3": -0.683, "KRT5": -0.872, "PKP1": -0.138,
                       "TESC": 0.597, "SPINK1": 0.768, "C9orf152": 0.534, "KRT7": 0.456}

# DNA hallmarks used for the label audit (RNA-independent evidence of lineage).
LUAD_DNA_HALLMARKS = ("mut_KRAS", "mut_EGFR", "mut_STK11", "amp_NKX2-1", "mut_BRAF", "mut_ALK", "mut_MET", "mut_RBM10")
LUSC_DNA_HALLMARKS = ("amp_SOX2", "amp_FGFR1", "mut_PTEN", "mut_KMT2D", "mut_NFE2L2", "amp_TP63")
