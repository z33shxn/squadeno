"""Part 3: the genes that drive the classification, and the pre-specified lipid module.

Drivers = the 20 genes that add most to the gap between the final model's mean score for LUSC and for LUAD.
For a linear model on standardised genes that gap splits exactly into one term per gene:
weight x (mean in LUSC - mean in LUAD), in standard deviations. Ranking by raw weight alone is also saved, as a
sensitivity check: it rewards genes that only adjust single samples and barely differ between the subtypes.
Two robustness checks: (i) bootstrap: how often each gene stays in the top 20 when the final pipeline is refitted
on 50 bootstrap resamples; (ii) L1 stability selection (100 half-subsamples): how often a sparse model selects it.
Cross-check: differential expression (Welch t-test on log2 CPM, Benjamini-Hochberg).
Lipid module: Akiyama 2017's corneocyte-lipid-envelope genes (pre-specified), tested across cohorts and
against each patient's own normal lung.
"""

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import io, models
from .analysis_model import design
from .config import AKIYAMA_CLE_GENES, LEMAY_GENES, RAW, RESULTS, SEEDS, SOLID_NORMAL


def _bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.clip(q, 0, 1)
    return out


def contributions(coef, Z, y):
    """Each gene's share of the gap between the mean model score of LUSC (y = 1) and LUAD (y = 0)."""
    return coef * (Z[y == 1].mean(axis=0) - Z[y == 0].mean(axis=0))


def _boot_top(Xa, y, C, genes, seed, k=20):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(y), len(y))
    m = models.logistic(C).fit(Xa[i], y[i])
    keep = m[0].keep_
    w = m[-1].coef_[0]
    gap = pd.Series(contributions(w, m[1].transform(Xa[i][:, keep]), y[i]), index=np.array(genes)[keep])
    by_gap = list(gap.sort_values(ascending=False).index[:k])
    by_weight = list(pd.Series(w, index=gap.index).abs().sort_values(ascending=False).index[:k])
    return by_gap, by_weight


def run():
    c, X, y, g = design()
    fm = json.load(open(RESULTS / "final_model.json"))
    C = fm["C"]
    coef = pd.Series(fm["coef"], index=fm["genes"])
    Z = (X.loc[fm["genes"]].T.values - np.array(fm["mean"])) / np.array(fm["scale"])
    gap = pd.Series(contributions(coef.values, Z, y), index=coef.index)
    top = gap.sort_values(ascending=False).index[:20]
    top_by_weight = coef.abs().sort_values(ascending=False).index[:20]
    Xa = X.T.values
    genes = list(X.index)

    boots = Parallel(n_jobs=8)(delayed(_boot_top)(Xa, y, C, genes, 1000 + b) for b in range(50))
    boot_freq = pd.Series([gg for b in boots for gg in b[0]]).value_counts() / len(boots)
    boot_freq_weight = pd.Series([gg for b in boots for gg in b[1]]).value_counts() / len(boots)

    expressed = X.index[((X.values > 1).mean(axis=1) > 0.2)]
    Xe = X.loc[expressed]
    Zs = StandardScaler().fit_transform(Xe.T.values)
    freq = pd.Series(0.0, index=expressed)
    n_sub = 0
    for s in SEEDS:
        rng = np.random.default_rng(s)
        for _ in range(20):
            idx = np.concatenate([rng.choice(np.where(y == k)[0], (y == k).sum() // 2, replace=False) for k in (0, 1)])
            l1 = LogisticRegression(C=0.05, l1_ratio=1.0, solver="liblinear", max_iter=5000, random_state=s).fit(Zs[idx], y[idx])
            freq[np.abs(l1.coef_[0]) > 0] += 1
            n_sub += 1
    freq /= n_sub

    counts = io.tcga_matrix("star_counts")[c.index]
    lib = np.power(2.0, counts).sub(1).sum(axis=0)
    cpm = np.log2(np.power(2.0, counts).sub(1).div(lib, axis=1) * 1e6 + 1)
    cpm = cpm.loc[cpm.index.intersection(expressed)]
    t, p = stats.ttest_ind(cpm.loc[:, y == 1], cpm.loc[:, y == 0], axis=1, equal_var=False)
    de = pd.DataFrame({"t": t, "p": p}, index=cpm.index)
    de["q"] = _bh(de.p.values)
    de["log2fc_cpm"] = cpm.loc[:, y == 1].mean(axis=1) - cpm.loc[:, y == 0].mean(axis=1)
    de.sort_values("p").to_csv(RESULTS / "differential_expression.tsv", sep="\t")

    probemap = pd.read_csv(RAW / "xena" / "gencode.v36.annotation.gtf.gene.probemap", sep="\t").drop_duplicates("gene").set_index("gene")
    cp, ycp, _ = io.cptac3()
    def auroc(gname):
        a = roc_auc_score(y, X.loc[gname].values)
        return float(max(a, 1 - a))

    rows = []
    for gname in top:
        v = X.loc[gname].values
        corr = pd.Series(np.corrcoef(np.vstack([v, Xe.values]))[0, 1:], index=expressed).drop(gname).sort_values(ascending=False)
        rows.append(dict(
            gene=gname, side="LUSC" if v[y == 1].mean() > v[y == 0].mean() else "LUAD",
            share_of_gap=float(gap[gname] / gap.sum()), coefficient_per_sd=float(coef[gname]),
            bootstrap_top20_freq=float(boot_freq.get(gname, 0.0)), l1_stability_freq=float(freq.get(gname, 0.0)),
            single_gene_auroc=auroc(gname), log2fc_lusc_vs_luad=float(v[y == 1].mean() - v[y == 0].mean()),
            de_q=float(de.q.get(gname, np.nan)), chrom=probemap.chrom.get(gname, ""),
            cptac3_auroc=float(max(roc_auc_score(ycp, cp.loc[gname]), 1 - roc_auc_score(ycp, cp.loc[gname]))) if gname in cp.index else np.nan,
            most_correlated=", ".join(f"{k} ({val:.2f})" for k, val in corr.head(3).items())))
    top20 = pd.DataFrame(rows)
    top20.insert(0, "rank", range(1, 21))
    top20.to_csv(RESULTS / "top20_genes.tsv", sep="\t", index=False)
    by_weight = pd.DataFrame([dict(rank=i + 1, gene=gname, coefficient_per_sd=float(coef[gname]), single_gene_auroc=auroc(gname),
                                   bootstrap_top20_freq=float(boot_freq_weight.get(gname, 0.0)), in_main_top20=gname in set(top))
                              for i, gname in enumerate(top_by_weight)])
    by_weight.to_csv(RESULTS / "top20_by_weight_sensitivity.tsv", sep="\t", index=False)

    # ---------------- lipid module (pre-specified; Akiyama 2017) ----------------
    allx = io.tcga_matrix("star_tpm")
    cohorts = {"TCGA": (X, y)}
    cohorts["CPTAC-3"] = (cp, ycp)
    xu, yu, _ = io.uppsala_rnaseq()
    cohorts["Uppsala RNA-seq (GSE81089)"] = (xu, yu)
    for acc in ("GSE30219", "GSE37745", "GSE50081", "GSE43580"):
        xa, ya, _ = io.geo_array(acc)
        cohorts[acc] = (xa, ya)

    def module_score(M, gs):
        gg = [x for x in gs if x in M.index]
        Zm = M.loc[gg]
        Zm = Zm.sub(Zm.mean(axis=1), axis=0).div(Zm.std(axis=1) + 1e-9, axis=0)
        return Zm.mean(axis=0).values

    lip = {}
    for name, (M, yy) in cohorts.items():
        row = {gg: (float(roc_auc_score(yy, M.loc[gg].values)) if gg in M.index else None) for gg in AKIYAMA_CLE_GENES + LEMAY_GENES}
        row["module_score"] = float(roc_auc_score(yy, module_score(M, AKIYAMA_CLE_GENES)))
        lip[name] = row
    lip_df = pd.DataFrame(lip)
    lip_df.to_csv(RESULTS / "lipid_module_auroc.tsv", sep="\t")

    normals = {col[:12]: col for col in allx.columns if col[13:15] == SOLID_NORMAL}
    paired = []
    for lab, name in ((1, "LUSC"), (0, "LUAD")):
        tum = [s for s in c.index[c.label == lab] if s[:12] in normals]
        nor = [normals[s[:12]] for s in tum]
        for gg in AKIYAMA_CLE_GENES + LEMAY_GENES:
            d = allx.loc[gg, tum].values - allx.loc[gg, nor].values
            paired.append(dict(tumour=name, gene=gg, n_pairs=len(tum), median_log2fc=float(np.median(d)), wilcoxon_p=float(stats.wilcoxon(d).pvalue)))
    paired = pd.DataFrame(paired)
    paired["bonferroni_p"] = (paired.wilcoxon_p * len(paired)).clip(upper=1)
    paired.to_csv(RESULTS / "lipid_paired_tumour_normal.tsv", sep="\t", index=False)
    L = X.loc[list(AKIYAMA_CLE_GENES), y == 1].T.rank().corr().values
    lusc_gained = paired[(paired.tumour == "LUSC") & paired.gene.isin(AKIYAMA_CLE_GENES) & (paired.median_log2fc > 0) & (paired.bonferroni_p < 0.05)].gene.tolist()

    res = dict(top20=top20.gene.tolist(), n_bootstrap=len(boots), n_l1_subsamples=n_sub, n_model_genes=len(coef),
               mean_score_gap=float(gap.sum()), top20_share_of_gap=float(top20.share_of_gap.sum()),
               top20_bootstrap_freq_median=float(top20.bootstrap_top20_freq.median()),
               by_weight_top20=by_weight.gene.tolist(), by_weight_bootstrap_freq_median=float(by_weight.bootstrap_top20_freq.median()),
               by_weight_overlap_with_top20=int(by_weight.in_main_top20.sum()),
               by_weight_top10_low_auroc=by_weight.head(10).query("single_gene_auroc < 0.6").gene.tolist(),
               top20_overlap_with_l1_top20=int(len(set(top20.gene) & set(freq.sort_values(ascending=False).index[:20]))),
               top20_max_de_q=float(top20.de_q.max()),
               lipid_module_auroc=lip_df.loc["module_score"].to_dict(), lipid_lusc_gained=lusc_gained,
               lipid_within_lusc_mean_rho=float(L[np.triu_indices(len(AKIYAMA_CLE_GENES), 1)].mean()),
               lipid_n_pairs={k: int(v) for k, v in paired.groupby("tumour").n_pairs.first().items()},
               cers3_lusc_log2fc=float(paired[(paired.tumour == "LUSC") & (paired.gene == "CERS3")].median_log2fc.iloc[0]),
               fa2h_luad_log2fc=float(paired[(paired.tumour == "LUAD") & (paired.gene == "FA2H")].median_log2fc.iloc[0]),
               fa2h_luad_bonf_p=float(paired[(paired.tumour == "LUAD") & (paired.gene == "FA2H")].bonferroni_p.iloc[0]),
               ugt8_lusc_log2fc=float(paired[(paired.tumour == "LUSC") & (paired.gene == "UGT8")].median_log2fc.iloc[0]),
               ugt8_luad_log2fc=float(paired[(paired.tumour == "LUAD") & (paired.gene == "UGT8")].median_log2fc.iloc[0]))
    json.dump(res, open(RESULTS / "genes.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=float))
