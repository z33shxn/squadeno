"""Part 4: could the classifier separate LUAD from LUSC for reasons other than tumour biology?

Each test states its hypothesis and the pass rule BEFORE looking at the result:
  T1 site batch       - if the model learned site, institution-grouped CV would score clearly below random CV.
                        Pass: drop in mean AUROC < 0.01.
  T2 non-biological   - a model using only sex, age, pack-years, tumour purity (DNA- and pathologist-based),
     covariates         RNA integrity, sequencing depth and tumour location. Pass: far below the expression model.
  T3 strata           - performance within each stratum of those covariates (>= 20 tumours, both classes).
                        Pass: AUROC stays >= 0.95 in every stratum.
  T4 treatment        - tumours excluded for treatment before resection, scored by the frozen model.
  T5 label noise      - DNA hallmarks of the other subtype should be enriched among tumours where RNA disagrees
                        with the label if the label (not the model) is wrong. Fisher's exact test.
  T6 tumour content   - low tumour content (mixing with normal lung) could bias calls; measured in real
                        low-purity tumours and by in-silico mixing with held-out TCGA normal lung.
"""

import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import io, models
from .analysis_model import design
from .cohort import excluded_tumours
from .config import LUAD_DNA_HALLMARKS, LUSC_DNA_HALLMARKS, PRIMARY_SEED, RESULTS, SOLID_NORMAL
from .metrics import summary


def covariates(c):
    d = pd.DataFrame(index=c.index)
    d["male"] = (c.sex == "male").astype(float)
    d["age"] = c.age
    d["pack_years"] = c.pack_years
    d["absolute_purity"] = c.absolute_purity
    d["path_tumour_nuclei"] = c.path_tumour_nuclei
    d["rin"] = c.rin
    d["log_library_size"] = np.log10(c.library_size)
    d["central_location"] = (c.location == "Central Lung").astype(float).where(c.location.notna())
    for col in list(d.columns):
        if d[col].isna().any():
            d[col + "_missing"] = d[col].isna().astype(float)
            d[col] = d[col].fillna(d[col].median())
    return d


def stratum_rows(y, p, mask_dict):
    rows = []
    for name, m in mask_dict.items():
        m = np.asarray(m, dtype=bool)
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        s = summary(y[m], p[m])
        rows.append(dict(stratum=name, n=s["n"], n_lusc=s["n_lusc"], auroc=s["auroc"], accuracy=s["accuracy"], lusc_recall=s["lusc_recall"], luad_recall=s["luad_recall"]))
    return rows


def run():
    c, X, y, g = design()
    pred = pd.read_csv(RESULTS / "cv_predictions.tsv", sep="\t", index_col=0).reindex(c.index)
    p = pred.p_lusc.values
    model = json.load(open(RESULTS / "model.json"))
    final = pickle.load(open(RESULTS / "final_model.pkl", "rb"))
    res = {}

    # T1 site batch
    ms = model["mean_sd"]
    res["T1_site_batch"] = dict(hypothesis="institution-grouped CV scores clearly below random CV", pass_rule="mean AUROC drop < 0.01",
                                random_auroc=ms["random"]["auroc"], institution_auroc=ms["institution"]["auroc"],
                                drop=model["delta_auroc_random_minus_institution"], passed=bool(model["delta_auroc_random_minus_institution"] < 0.01),
                                tss_code_only_accuracy=float(c.groupby("tss").label.agg(lambda s: max(s.mean(), 1 - s.mean())).mul(c.groupby("tss").size()).sum() / len(c)))

    # T2 covariate-only model
    cov = covariates(c)
    pc = np.zeros(len(y))
    for tr, te in models.folds("institution", cov.values, y, g, PRIMARY_SEED):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced")).fit(cov.values[tr], y[tr])
        pc[te] = m.predict_proba(cov.values[te])[:, 1]
    single = {col: float(max(roc_auc_score(y, cov[col]), 1 - roc_auc_score(y, cov[col]))) for col in cov.columns if not col.endswith("_missing")}
    res["T2_covariates_only"] = dict(hypothesis="non-biological covariates alone reproduce the separation", pass_rule="covariate model AUROC far below expression model",
                                     covariate_model=summary(y, pc), single_covariate_auroc=single, expression_model_auroc=summary(y, p)["auroc"])

    # T3 strata
    pur_t = c.absolute_purity.quantile([1 / 3, 2 / 3]).values
    rin_m = c.rin.median()
    lib_t = c.library_size.quantile([1 / 3, 2 / 3]).values
    strata = {
        "female": c.sex == "female", "male": c.sex == "male",
        "never smoker": c.smoking_indicator == 1, "ever smoker": c.smoking_indicator.isin([2, 3, 4, 5]),
        "ABSOLUTE purity low tertile": c.absolute_purity < pur_t[0], "ABSOLUTE purity mid tertile": (c.absolute_purity >= pur_t[0]) & (c.absolute_purity < pur_t[1]),
        "ABSOLUTE purity high tertile": c.absolute_purity >= pur_t[1],
        "pathologist tumour nuclei below 60%": c.path_tumour_nuclei < 60, "pathologist tumour nuclei 60% or more": c.path_tumour_nuclei >= 60,
        "RIN below median": c.rin < rin_m, "RIN at or above median": c.rin >= rin_m,
        "depth low tertile": c.library_size < lib_t[0], "depth high tertile": c.library_size >= lib_t[1],
        "central tumour": c.location == "Central Lung", "peripheral tumour": c.location == "Peripheral Lung",
        "stage I": c.stage.astype(str).str.match(r"^Stage IA?B?$|^Stage I$"), "stage II to IV": c.stage.astype(str).str.match(r"^Stage I[IV]"),
        "institution shared by both types": c.groupby("institution").label.transform("nunique") > 1,
    }
    res["T3_strata"] = stratum_rows(y, p, {k: v.values for k, v in strata.items()})

    # T4 treatment-related exclusions, scored by the frozen model (they were never used for training)
    allx = io.tcga_matrix("star_tpm")
    clin = io.tcga_clinical()
    scored = {}
    for name, key in (("neoadjuvant-treated", "neoadjuvant"), ("prior treatment for another cancer (Do_not_use)", "Do_not_use"), ("AWG pathology exclusions (NSCLC-NOS/LCNEC)", "AWG")):
        idx = [s for s in excluded_tumours(key) if s in allx.columns]
        if not idx:
            continue
        pe = final["model"].predict_proba(allx.loc[final["genes"], idx].T.values.astype("float64"))[:, 1]
        lab = (clin.project.reindex(idx) == "LUSC").astype(int).values
        scored[name] = dict(n=len(idx), accuracy_vs_label=float(((pe > 0.5) == lab).mean()), called_lusc=float((pe > 0.5).mean()))
    res["T4_treatment_and_exclusions"] = dict(cv_error_rate=float(((p > 0.5) != y).mean()), scored=scored)

    # T5 label noise: DNA hallmarks
    dna = io.dna_panel()
    ann = io.gdc_annotations()
    bad_dna = set(ann.loc[ann.category == "Item flagged DNU", "entity_submitter_id"].astype(str).str[:15])
    d = pd.DataFrame({"label": y, "p": p}, index=c.index)
    d["s15"] = d.index.str[:15]
    d = d[d.s15.isin(dna.index) & ~d.s15.isin(bad_dna)]
    f = dna.reindex(d.s15.values)
    d["luad_hallmark"] = f[[h for h in LUAD_DNA_HALLMARKS if h in f.columns]].sum(axis=1).values > 0
    d["lusc_hallmark"] = f[[h for h in LUSC_DNA_HALLMARKS if h in f.columns]].sum(axis=1).values > 0
    d["disputed"] = (d.p > 0.5) != (d.label == 1)
    t5 = {}
    for lab, hall, name in ((1, "luad_hallmark", "LUSC-labelled"), (0, "lusc_hallmark", "LUAD-labelled")):
        m = d.label == lab
        tab = [[int((d[m & d.disputed][hall]).sum()), int((~d[m & d.disputed][hall]).sum())],
               [int((d[m & ~d.disputed][hall]).sum()), int((~d[m & ~d.disputed][hall]).sum())]]
        odds, pv = fisher_exact(tab)
        t5[name] = dict(disputed_with_other_type_hallmark=tab[0], concordant_with_other_type_hallmark=tab[1], odds_ratio=float(odds), fisher_p=float(pv))
    d[d.disputed].to_csv(RESULTS / "disputed_labels.tsv", sep="\t")
    res["T5_label_noise_dna"] = dict(hypothesis="if labels (not the model) are wrong, DNA hallmarks of the other subtype are enriched in disputed tumours",
                                     n_tumours_with_dna=int(len(d)), n_excluded_failed_dna_qc=int(c.index.str[:15].isin(bad_dna).sum()),
                                     n_disputed=int(d.disputed.sum()), **t5)

    # T6 tumour content
    low = c.absolute_purity < 0.25
    t6 = dict(real_low_purity=dict(n=int(low.sum()), lusc_recall=float(((p > 0.5) & (y == 1) & low).sum() / max(1, ((y == 1) & low).sum())),
                                   luad_recall=float(((p <= 0.5) & (y == 0) & low).sum() / max(1, ((y == 0) & low).sum()))),
              real_higher_purity=dict(n=int((c.absolute_purity >= 0.25).sum()),
                                      lusc_recall=float(((p > 0.5) & (y == 1) & ~low & c.absolute_purity.notna()).sum() / max(1, ((y == 1) & ~low & c.absolute_purity.notna()).sum())),
                                      luad_recall=float(((p <= 0.5) & (y == 0) & ~low & c.absolute_purity.notna()).sum() / max(1, ((y == 0) & ~low & c.absolute_purity.notna()).sum()))))
    normals = [col for col in allx.columns if col[13:15] == SOLID_NORMAL]
    N = allx.loc[X.index, normals].values.astype("float64")
    norm_pat = np.array([n[:12] for n in normals])
    rng = np.random.default_rng(PRIMARY_SEED)
    lin = lambda a: np.power(2.0, a) - 1  # noqa: E731
    fold = pred.fold.values
    dil = []
    for frac in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05):
        pp = np.zeros(len(y))
        for k in range(fold.max() + 1):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            m = models.logistic(final["C"]).fit(X.values[:, tr].T, y[tr])
            ok = ~np.isin(norm_pat, c.patient.values[tr])  # never mix in a normal from a training patient
            pick = rng.choice(np.where(ok)[0], len(te))
            mix = np.log2(frac * lin(X.values[:, te]) + (1 - frac) * lin(N[:, pick]) + 1)
            pp[te] = m.predict_proba(mix.T)[:, 1]
        s = summary(y, pp)
        dil.append(dict(tumour_fraction=frac, auroc=s["auroc"], accuracy=s["accuracy"], lusc_recall=s["lusc_recall"], luad_recall=s["luad_recall"]))
    t6["in_silico_dilution"] = dil
    res["T6_tumour_content"] = t6
    json.dump(res, open(RESULTS / "confounders.json", "w"), indent=1, default=float)
    pd.DataFrame(res["T3_strata"]).to_csv(RESULTS / "confounder_strata.tsv", sep="\t", index=False)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=float))
