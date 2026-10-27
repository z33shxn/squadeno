"""Part 1: which samples, and why.

Starts from every RNA-seq sample in the Xena GDC-hub LUAD and LUSC matrices and applies, in order:
  1. primary solid tumour only (barcode sample type 01; drops normals 11 and recurrences 02)
  2. drop FFPE samples (degraded RNA; those patients also have a frozen sample)
  3. drop TCGA Analysis Working Group pathology exclusions (re-reviewed as NSCLC-NOS or LCNEC)
  4. drop PanCanAtlas RNA-seq `Do_not_use` (unacceptable prior treatment for another cancer)
  5. drop tumours treated before resection (GDC annotation "Neoadjuvant therapy", Xena prior_treatment,
     or legacy HISTORY_NEOADJUVANT_TRTYN)
  6. drop cases GDC flags as a recurrence submitted as primary, or with molecular results off specification
  7. one sample per patient (vial A preferred)
Outputs results/cohort.tsv, results/excluded_samples.tsv and results/cohort_waterfall.tsv.
"""

import numpy as np
import pandas as pd

from . import io
from .config import EXCLUDING_ANNOTATIONS, INSTITUTION_ALIASES, PRIMARY_TUMOUR, RESULTS


def _institution(name: str) -> str:
    name = str(name).strip()
    return INSTITUTION_ALIASES.get(name, name)


def build_cohort(write: bool = True):
    tpm = io.tcga_matrix("star_tpm")
    clin = io.tcga_clinical()
    samples = pd.DataFrame(index=pd.Index(tpm.columns, name="sample"))
    parts = samples.index.to_series().apply(io.barcode_parts).apply(pd.Series)
    s = samples.join(parts)
    s["project"] = clin["project"].reindex(s.index)
    s["is_ffpe"] = clin["is_ffpe.samples"].reindex(s.index).astype(str).str.lower().eq("true")

    q = io.pancan_quality()
    awg = set(q.loc[q.AWG_excluded_because_of_pathology == 1, "patient_barcode"])
    rna_q = q[q.platform.str.contains("RNASeq", na=False)]
    dnu = set(rna_q.loc[rna_q.Do_not_use == True, "patient_barcode"])  # noqa: E712
    ann = io.gdc_annotations()
    neo_ann = set(ann.loc[ann.category == "Neoadjuvant therapy", "case_submitter_id"])
    other_ann = set(ann.loc[ann.category.isin([c for c in EXCLUDING_ANNOTATIONS if c != "Neoadjuvant therapy"]), "case_submitter_id"])
    prior_tx = set(clin.index[clin["prior_treatment.diagnoses"].astype(str).str.lower().eq("yes")].str[:12])
    legacy = io.legacy_clinical()
    legacy_neo = set(legacy.index[legacy.get("HISTORY_NEOADJUVANT_TRTYN", pd.Series(dtype=str)).astype(str).str.startswith("Yes")])
    neo = neo_ann | prior_tx | legacy_neo

    steps = [
        ("primary tumour only (sample type 01)", lambda d: d.sample_type == PRIMARY_TUMOUR),
        ("drop FFPE samples", lambda d: ~d.is_ffpe),
        ("drop AWG pathology exclusions (NSCLC-NOS / favour LCNEC)", lambda d: ~d.patient.isin(awg)),
        ("drop PanCanAtlas Do_not_use (prior treatment for another cancer)", lambda d: ~d.patient.isin(dnu)),
        ("drop tumours treated before resection (neoadjuvant)", lambda d: ~d.patient.isin(neo)),
        ("drop GDC-flagged recurrence / off-spec cases", lambda d: ~d.patient.isin(other_ann)),
    ]
    waterfall = [("all Xena RNA-seq samples", (s.project == "LUAD").sum(), (s.project == "LUSC").sum())]
    excluded = []
    cur = s
    for name, keep in steps:
        k = keep(cur)
        excluded.append(cur[~k].assign(reason=name))
        cur = cur[k]
        waterfall.append((name, (cur.project == "LUAD").sum(), (cur.project == "LUSC").sum()))
    cur = cur.sort_values(["patient", "vial"])
    dup = cur.patient.duplicated(keep="first")
    excluded.append(cur[dup].assign(reason="second sample of the same patient"))
    cur = cur[~dup]
    waterfall.append(("one sample per patient", (cur.project == "LUAD").sum(), (cur.project == "LUSC").sum()))

    # covariates used later (confounder tests); never used as model features
    c = cur.copy()
    c["label"] = (c.project == "LUSC").astype(int)
    c["institution"] = clin["name.tissue_source_site"].reindex(c.index).map(_institution)
    al = io.gdc_aliquots()
    al = al[al["sample"].isin(c.index)].sort_values("aliquot")
    c["n_rna_aliquots"] = al.groupby("sample").size().reindex(c.index).fillna(0).astype(int)
    c["plate"] = al.drop_duplicates("sample").set_index("sample").plate.reindex(c.index)
    c["sex"] = clin["gender.demographic"].reindex(c.index)
    c["age"] = pd.to_numeric(clin["age_at_index.demographic"].reindex(c.index), errors="coerce")
    c["race"] = clin["race.demographic"].reindex(c.index)
    c["pack_years"] = pd.to_numeric(clin["pack_years_smoked.exposures"].reindex(c.index), errors="coerce")
    c["smoking_indicator"] = pd.to_numeric(legacy["TOBACCO_SMOKING_HISTORY_INDICATOR"].reindex(c.patient).values, errors="coerce")
    c["location"] = legacy["LOCATION_LUNG_PARENCHYMA"].reindex(c.patient).values
    c["stage"] = clin["ajcc_pathologic_stage.diagnoses"].reindex(c.index)
    c["diagnosis"] = clin["primary_diagnosis.diagnoses"].reindex(c.index)
    c["absolute_purity"] = io.absolute_purity().reindex(c.index.str[:15]).values
    path = io.slide_pathology().reindex(c.index)
    c["path_tumour_nuclei"] = path.percent_tumor_nuclei.values
    c["path_necrosis"] = path.percent_necrosis.values
    c["rin"] = io.rna_integrity().reindex(c.index).values
    c["library_size"] = io.tcga_library_size().reindex(c.index).values

    assert c.patient.is_unique, "a patient appears twice"
    assert (c.sample_type == PRIMARY_TUMOUR).all()
    assert not c.is_ffpe.any()
    assert not c.patient.isin(awg | dnu | neo | other_ann).any()
    assert c.institution.notna().all(), "every tumour needs an institution for the grouped split"

    wf = pd.DataFrame(waterfall, columns=["step", "LUAD", "LUSC"])
    wf["total"] = wf.LUAD + wf.LUSC
    ex = pd.concat(excluded)[["patient", "project", "sample_type", "vial", "reason"]]
    if write:
        c.to_csv(RESULTS / "cohort.tsv", sep="\t")
        ex.to_csv(RESULTS / "excluded_samples.tsv", sep="\t")
        wf.to_csv(RESULTS / "cohort_waterfall.tsv", sep="\t", index=False)
    stats = dict(n_neoadjuvant_sources=dict(gdc_annotation=len(neo_ann), xena_prior_treatment=len(prior_tx), legacy_flag=len(legacy_neo)),
                 n_tss_codes=int(c.tss.nunique()), n_institutions=int(c.institution.nunique()),
                 n_shared_tss_codes=int(c.groupby("tss").label.nunique().gt(1).sum()),
                 n_shared_institutions=int(c.groupby("institution").label.nunique().gt(1).sum()),
                 n_plates=int(c.plate.nunique()), n_shared_plates=int(c.groupby("plate").label.nunique().gt(1).sum()),
                 n_double_sequenced=int((c.n_rna_aliquots > 1).sum()))
    return c, ex, wf, stats


def load_cohort() -> pd.DataFrame:
    p = RESULTS / "cohort.tsv"
    if not p.exists():
        build_cohort()
    return pd.read_csv(p, sep="\t", index_col=0)


def excluded_tumours(reason_contains: str) -> pd.Index:
    ex = pd.read_csv(RESULTS / "excluded_samples.tsv", sep="\t", index_col=0)
    ex = ex[(ex.sample_type.astype(str).str.zfill(2) == PRIMARY_TUMOUR) & ex.reason.str.contains(reason_contains, regex=False)]
    return ex.index


if __name__ == "__main__":
    c, ex, wf, st = build_cohort()
    print(wf.to_string(index=False))
    print(st)
    print(np.round(c.groupby("project")[["absolute_purity", "path_tumour_nuclei", "rin"]].median(), 2))
