"""External validation of the FROZEN TCGA model and rank rules, and a head-to-head benchmark against two
published classifiers (Li 2019 KRT5>AGR2; Hamaneh 2022 8-gene signature) on identical cohorts.

The L2 model needs log2(TPM+1) RNA-seq, so it is applied to CPTAC-3 (same GDC pipeline), Uppsala RNA-seq
(FPKM converted to TPM) and the Korean cohort; the rank rules and both baselines use only within-sample
comparisons or within-cohort z-scores, so they also run on the Affymetrix cohorts.
"""

import json

import pandas as pd

from . import io, models
from .cohort import load_cohort
from .config import RESULTS
from .metrics import bootstrap_ci, summary


def _l2(final, M):
    """Frozen model scores. Model genes missing from M are set to their TCGA training mean (they add nothing)."""
    missing = [g for g in final["genes"] if g not in M.index]
    return models.predict_from_json(final, M), len(missing)


def run():
    final = json.load(open(RESULTS / "final_model.json"))
    rules = [(r["lusc_gene"], r["luad_gene"], r["training_score"]) for r in json.load(open(RESULTS / "rank_rules.json"))]
    c = load_cohort()
    cv = pd.read_csv(RESULTS / "cv_predictions.tsv", sep="\t", index_col=0).reindex(c.index)
    X = io.tcga_matrix("star_tpm")[c.index]
    y = c.label.values

    cohorts = {}
    cp, ycp, sel = io.cptac3()
    cohorts["CPTAC-3"] = ("rnaseq", cp, ycp)
    xu, yu, _ = io.uppsala_rnaseq()
    cohorts["Uppsala RNA-seq (GSE81089)"] = ("rnaseq", xu, yu)
    for acc, desc in (("GSE30219", "France"), ("GSE37745", "Sweden"), ("GSE50081", "Canada"), ("GSE43580", "SBV IMPROVER")):
        xa, ya, _ = io.geo_array(acc)
        cohorts[f"{acc} ({desc})"] = ("array", xa, ya)

    rows = []
    # TCGA itself: cross-validated predictions (institution-grouped), not the in-sample fit
    for method, pp in (("L2 logistic (ours)", cv.p_lusc.values), ("9 rank rules (ours)", cv.p_rules.values),
                       ("Li 2019 KRT5>AGR2", models.li2019(X)), ("Hamaneh 2022 8-gene", models.hamaneh2022(X)[0])):
        rows.append(dict(cohort="TCGA (CV, institution-grouped)", platform="RNA-seq", method=method, **summary(y, pp)))
    notes = {}
    for name, (plat, M, yy) in cohorts.items():
        if plat == "rnaseq":
            pp, miss = _l2(final, M)
            notes[name] = dict(l2_genes_missing=miss)
            rows.append(dict(cohort=name, platform="RNA-seq", method="L2 logistic (ours)", **summary(yy, pp), auroc_ci95=bootstrap_ci(yy, pp)))
        pr, nr = models.predict_rank_rules(M, rules)
        rows.append(dict(cohort=name, platform="RNA-seq" if plat == "rnaseq" else "Affymetrix", method="9 rank rules (ours)", rules_used=nr, **summary(yy, pr)))
        rows.append(dict(cohort=name, platform="RNA-seq" if plat == "rnaseq" else "Affymetrix", method="Li 2019 KRT5>AGR2", **summary(yy, models.li2019(M))))
        ph, ng = models.hamaneh2022(M)
        rows.append(dict(cohort=name, platform="RNA-seq" if plat == "rnaseq" else "Affymetrix", method="Hamaneh 2022 8-gene", genes_used=ng, **summary(yy, ph)))
    bench = pd.DataFrame(rows)
    bench.to_csv(RESULTS / "external_benchmark.tsv", sep="\t", index=False)
    mean = bench.groupby("method")[["auroc", "accuracy", "lusc_recall", "luad_recall"]].mean()
    mean.to_csv(RESULTS / "external_benchmark_mean.tsv", sep="\t")

    # CPTAC-3 ancestry subgroups (TCGA is ~2% Asian; CPTAC-3 lung is ~39% Asian)
    pcp, _ = _l2(final, cp)
    asian = (sel.race == "asian").values
    sub = {"Asian": summary(ycp[asian], pcp[asian]), "non-Asian": summary(ycp[~asian], pcp[~asian])}
    prc, _ = models.predict_rank_rules(cp, rules)
    sub_rules = {"Asian": summary(ycp[asian], prc[asian]), "non-Asian": summary(ycp[~asian], prc[~asian])}

    # Korean cohort (all LUAD + adjacent normal lung): specificity and the normal-lung check
    xk, is_normal, mk = io.korean_luad()
    pk, missk = _l2(final, xk)
    prk, _ = models.predict_rank_rules(xk, rules)
    never = mk.smoking_status.astype(str).str.contains("never").values
    korea = dict(n_luad=int((~is_normal).sum()), n_normal=int(is_normal.sum()), l2_genes_missing=missk,
                 luad_called_luad_l2=float((pk[~is_normal] < 0.5).mean()), luad_called_luad_rules=float((prk[~is_normal] < 0.5).mean()),
                 never_smoker_luad_n=int((never & ~is_normal).sum()), never_smoker_luad_called_luad_l2=float((pk[never & ~is_normal] < 0.5).mean()),
                 normal_lung_called_luad_l2=float((pk[is_normal] < 0.5).mean()))
    tcga_asian = int((c.race == "asian").sum())
    res = dict(benchmark_mean=mean.round(4).to_dict(orient="index"), notes=notes, cptac3_ancestry_l2=sub, cptac3_ancestry_rules=sub_rules,
               cptac3_n=int(len(ycp)), cptac3_n_asian=int(asian.sum()), tcga_n_asian=tcga_asian, korea=korea)
    json.dump(res, open(RESULTS / "external.json", "w"), indent=1, default=float)
    return res, bench


if __name__ == "__main__":
    r, b = run()
    print(b.round(3).to_string(index=False))
    print(json.dumps(r, indent=1, default=float))
