"""Part 2: model, defensible split, and leakage audit.

Primary model: L2-penalised logistic regression on log2(TPM+1) of protein-coding genes (expression filter and
scaling fitted inside each training fold; class_weight='balanced'; C tuned by inner CV).
Split: outer 5-fold StratifiedGroupKFold grouped by INSTITUTION (tissue-source sites mapped to institutions),
repeated over five seeds. Patients are unique, so a patient can never be in train and test.
Why: TCGA issued tissue-source-site codes per institution per disease, so a site code is a perfect label proxy;
grouping by institution stops the model from being rewarded for recognising site batch.
"""

import json
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import io, models
from .cohort import load_cohort
from .config import PRIMARY_SEED, RESULTS, SEEDS
from .metrics import bootstrap_ci, summary


def design():
    c = load_cohort()
    pc = io.protein_coding_symbols()
    X = io.tcga_matrix("star_tpm")
    X = X.loc[[g for g in X.index if g in pc], c.index].astype("float64")
    return c, X, c.label.values, c.institution.values


def _run_seed(Xa, y, g, seed, kind):
    p, fold, Cs = models.nested_cv(Xa, y, g, seed, kind)
    return seed, kind, p, fold, Cs


def random_label_canary(Xa, y, groups, seed=PRIMARY_SEED, k_features=50):
    """Shuffle labels: an honest pipeline must score ~0.5. Selecting genes on ALL samples before splitting
    (a classic leak) is shown for contrast."""
    rng = np.random.default_rng(seed)
    yr = rng.permutation(y)
    var = Xa.var(axis=0)
    Xv = Xa[:, np.argsort(var)[-5000:]]
    out = {}
    for name, leaky in (("selection inside folds (our pipeline)", False), ("selection before split (leaky)", True)):
        Xs = SelectKBest(f_classif, k=k_features).fit_transform(Xv, yr) if leaky else Xv
        p = np.zeros(len(yr))
        for tr, te in models.folds("institution", Xs, yr, groups, seed):
            steps = [StandardScaler()] + ([] if leaky else [SelectKBest(f_classif, k=k_features)]) + [LogisticRegression(C=0.1, max_iter=5000)]
            p[te] = make_pipeline(*steps).fit(Xs[tr], yr[tr]).predict_proba(Xs[te])[:, 1]
        out[name] = float(roc_auc_score(yr, p))
    return out


def nearest_neighbour_audit(Xa, y, groups, inst, seed=PRIMARY_SEED, n_genes=2000):
    """Residual-similarity audit: for each test tumour, the highest Pearson correlation to any training tumour,
    and whether that nearest neighbour came from the same institution, under random vs institution splits."""
    v = np.argsort(Xa.var(axis=0))[-n_genes:]
    Z = Xa[:, v]
    Z = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(axis=1, keepdims=True)
    R = Z @ Z.T / Z.shape[1]
    res = {}
    for kind in ("random", "institution"):
        top, same = [], []
        for tr, te in models.folds(kind, Xa, y, groups, seed):
            sub = R[np.ix_(te, tr)]
            j = sub.argmax(axis=1)
            top.extend(sub.max(axis=1))
            same.extend(inst[tr][j] == inst[te])
        top = np.array(top)
        res[kind] = dict(median_max_r=float(np.median(top)), frac_r_over_0_9=float((top > 0.9).mean()),
                         frac_nearest_from_same_institution=float(np.mean(same)))
    return res


def trustworthiness(y, p):
    conf = np.abs(p - 0.5)
    order = np.argsort(-conf)
    rows = []
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        k = order[: int(round(cov * len(y)))]
        rows.append(dict(coverage=cov, n=int(len(k)), accuracy=float(((p[k] > 0.5) == y[k]).mean()), auroc=float(roc_auc_score(y[k], p[k]))))
    return rows


def run():
    c, X, y, g = design()
    Xa = X.T.values
    jobs = [(s, k) for s in SEEDS for k in ("institution", "random")]
    out = Parallel(n_jobs=min(10, len(jobs)), verbose=0)(delayed(_run_seed)(Xa, y, g, s, k) for s, k in jobs)
    preds, rows, chosen = {}, [], {}
    for seed, kind, p, fold, Cs in out:
        preds[(seed, kind)] = (p, fold)
        rows.append(dict(seed=seed, split=kind, **summary(y, p)))
        chosen[f"{kind}_{seed}"] = Cs
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS / "model_cv_per_seed.tsv", sep="\t", index=False)
    agg = per_seed.groupby("split")[["auroc", "auprc", "accuracy", "balanced_accuracy", "lusc_recall", "luad_recall", "ece", "brier"]].agg(["mean", "std"])
    agg.to_csv(RESULTS / "model_cv_summary.tsv", sep="\t")

    p42, fold42 = preds[(PRIMARY_SEED, "institution")]
    pred = c[["patient", "project", "institution"]].copy()
    pred["p_lusc"], pred["fold"] = p42, fold42
    for s in SEEDS:
        pred[f"p_lusc_seed{s}"] = preds[(s, "institution")][0]
    pred["p_lusc_mean5"] = pred[[f"p_lusc_seed{s}" for s in SEEDS]].mean(axis=1)
    pred.to_csv(RESULTS / "cv_predictions.tsv", sep="\t")

    # interpretable rank rules, evaluated in the same institution folds (fit on training folds only)
    Xdf = X
    pr = np.zeros(len(y))
    for k in range(fold42.max() + 1):
        tr, te = np.where(fold42 != k)[0], np.where(fold42 == k)[0]
        rules = models.fit_rank_rules(Xdf.iloc[:, tr], y[tr])
        pr[te] = models.predict_rank_rules(Xdf.iloc[:, te], rules)[0]
    pred["p_rules"] = pr
    pred.to_csv(RESULTS / "cv_predictions.tsv", sep="\t")
    final_rules = models.fit_rank_rules(Xdf, y)
    models.save_rules(final_rules, RESULTS / "rank_rules.json")

    # final frozen model for external cohorts: C chosen by institution-grouped CV on the whole cohort
    C_final, c_scores = models.choose_C(Xa, y, g, PRIMARY_SEED)
    final = models.logistic(C_final).fit(Xa, y)
    genes_used = np.array(X.index)[final[0].keep_]
    coef = pd.Series(final[-1].coef_[0], index=genes_used).sort_values()
    coef.to_csv(RESULTS / "final_model_coefficients.tsv", sep="\t", header=["coefficient_per_sd"])
    with open(RESULTS / "final_model.pkl", "wb") as f:
        pickle.dump({"model": final, "genes": list(X.index), "C": C_final}, f)
    scaler, lr = final[1], final[2]
    json.dump(dict(description="L2 logistic regression on log2(TPM+1); P(LUSC) = sigmoid(intercept + sum(coef * (x - mean) / scale))",
                   C=C_final, intercept=float(lr.intercept_[0]), genes=list(genes_used),
                   mean=[float(v) for v in scaler.mean_], scale=[float(v) for v in scaler.scale_], coef=[float(v) for v in lr.coef_[0]]),
              open(RESULTS / "final_model.json", "w"))

    res = dict(
        n=int(len(y)), n_luad=int((y == 0).sum()), n_lusc=int(y.sum()), n_genes_protein_coding=int(X.shape[0]),
        seeds=list(SEEDS), C_grid_scores_full=c_scores, C_final=C_final, chosen_C=chosen,
        institution=summary(y, p42), institution_mean5=summary(y, pred.p_lusc_mean5.values),
        auroc_ci95_institution=bootstrap_ci(y, p42), auprc_ci95_institution=bootstrap_ci(y, p42, fn=lambda a, b: __import__("sklearn.metrics").metrics.average_precision_score(a, b)),
        mean_sd={k: {m: [float(agg.loc[k, (m, "mean")]), float(agg.loc[k, (m, "std")])] for m in ("auroc", "auprc", "accuracy", "balanced_accuracy", "ece", "brier")} for k in ("institution", "random")},
        delta_auroc_random_minus_institution=float(agg.loc["random", ("auroc", "mean")] - agg.loc["institution", ("auroc", "mean")]),
        rank_rules_cv=summary(y, pr), rank_rules=[{"lusc_gene": a, "luad_gene": b} for a, b, _ in final_rules],
        random_label_canary=random_label_canary(Xa, y, g),
        nearest_neighbour_audit=nearest_neighbour_audit(Xa, y, g, c.institution.values),
        trustworthiness=trustworthiness(y, p42),
        n_errors_seed42=int(((p42 > 0.5) != y).sum()),
    )
    json.dump(res, open(RESULTS / "model.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("institution", "mean_sd", "delta_auroc_random_minus_institution", "rank_rules_cv", "random_label_canary", "nearest_neighbour_audit")}, indent=1))
