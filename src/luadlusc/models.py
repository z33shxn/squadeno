"""Models: the primary penalised logistic regression, the interpretable rank rules, and published baselines."""

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import C_GRID, EXPR_MIN_FRACTION, EXPR_THRESHOLD, HAMANEH2022_WEIGHTS, LI2019_RULE, N_INNER, N_OUTER


class ExpressedGenes(BaseEstimator, TransformerMixin):
    """Keep genes expressed (> threshold) in more than `min_fraction` of TRAINING samples. Fit inside each fold."""

    def __init__(self, threshold=EXPR_THRESHOLD, min_fraction=EXPR_MIN_FRACTION):
        self.threshold, self.min_fraction = threshold, min_fraction

    def fit(self, X, y=None):
        self.keep_ = (np.asarray(X) > self.threshold).mean(axis=0) > self.min_fraction
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def logistic(C):
    return make_pipeline(ExpressedGenes(), StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


def splitter(kind, n, seed):
    if kind == "institution":
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    if kind == "random":
        return StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    raise ValueError(kind)


def folds(kind, X, y, groups, seed, n=N_OUTER):
    sp = splitter(kind, n, seed)
    return list(sp.split(X, y, groups if kind == "institution" else None))


def choose_C(X, y, groups, seed, kind="institution"):
    """Inner CV (same split type as the outer loop) over C_GRID; returns the C with the best mean AUROC."""
    scores = {}
    for C in C_GRID:
        s = []
        for tr, te in folds(kind, X, y, groups, seed, n=N_INNER):
            m = logistic(C).fit(X[tr], y[tr])
            s.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        scores[C] = float(np.mean(s))
    return max(scores, key=scores.get), scores


def nested_cv(X, y, groups, seed, kind="institution"):
    """Outer CV predictions with C tuned inside each training fold. Returns (p, fold_id, chosen_Cs)."""
    p = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    chosen = []
    for k, (tr, te) in enumerate(folds(kind, X, y, groups, seed)):
        C, _ = choose_C(X[tr], y[tr], groups[tr], seed, kind)
        m = logistic(C).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        fold_id[te] = k
        chosen.append(C)
    assert not np.isnan(p).any()
    return p, fold_id, chosen


# ------------------------------------------------------------------ rank rules (k top-scoring pairs)

def fit_rank_rules(X: pd.DataFrame, y, k=9, n_candidates=80, min_median=4.0, exclude=()):
    """Learn k disjoint gene pairs (a, b): vote LUSC when x[a] > x[b] within a sample.

    Candidates are the genes with the largest single-gene separation in the TRAINING data, restricted to genes
    expressed at log2(TPM+1) >= min_median in the class they mark. The expression floor keeps rules away from
    comparisons between two near-zero genes, which are unstable on other platforms (microarrays).
    """
    y = np.asarray(y).astype(bool)
    auc = X.apply(lambda r: roc_auc_score(y, r.values), axis=1)
    med1, med0 = X.loc[:, y].median(axis=1), X.loc[:, ~y].median(axis=1)
    sq = [g for g in auc.sort_values(ascending=False).index if med1[g] >= min_median and auc[g] > 0.85 and g not in exclude][:n_candidates]
    ad = [g for g in auc.sort_values().index if med0[g] >= min_median and auc[g] < 0.15 and g not in exclude][:n_candidates]
    A, B = X.loc[sq].values, X.loc[ad].values
    gt = A[:, None, :] > B[None, :, :]
    score = gt[:, :, y].mean(axis=2) - gt[:, :, ~y].mean(axis=2)
    pairs, ua, ub = [], set(), set()
    for flat in np.argsort(-score, axis=None):
        i, j = np.unravel_index(flat, score.shape)
        if i in ua or j in ub:
            continue
        pairs.append((sq[i], ad[j], float(score[i, j])))
        ua.add(i)
        ub.add(j)
        if len(pairs) == k:
            break
    return pairs


def predict_rank_rules(X: pd.DataFrame, pairs):
    """Fraction of usable rules voting LUSC (rules whose genes are absent from X are skipped)."""
    votes = [(X.loc[a] > X.loc[b]).astype(float).values for a, b, _ in pairs if a in X.index and b in X.index]
    if not votes:
        raise ValueError("no rule genes present")
    return np.mean(votes, axis=0), len(votes)


def save_rules(pairs, path):
    json.dump([{"lusc_gene": a, "luad_gene": b, "training_score": round(s, 4)} for a, b, s in pairs], open(path, "w"), indent=1)


# ------------------------------------------------------------------ published baselines

def li2019(X: pd.DataFrame):
    """Li et al. 2019 (PMID 31752667): LUSC if KRT5 > AGR2."""
    a, b = LI2019_RULE
    return (X.loc[a] > X.loc[b]).astype(float).values


def hamaneh2022(X: pd.DataFrame):
    """Hamaneh et al. 2022 (PMID 35722224): correlation of the within-cohort z-scored 8-gene profile with the
    published weight vector; returned on a 0-1 scale where > 0.5 means LUSC."""
    w = pd.Series(HAMANEH2022_WEIGHTS)
    g = [x for x in w.index if x in X.index]
    Z = X.loc[g]
    Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1) + 1e-9, axis=0)
    r = Z.apply(lambda c: np.corrcoef(c.values, w[g].values)[0, 1])
    return ((1 - r) / 2).values, len(g)


def predict_from_json(model_json: dict, X: pd.DataFrame):
    """Score samples with the exported frozen model (no pickle needed). X: log2(TPM+1), genes x samples.
    A gene missing from X is set to its training mean, so it adds nothing to the score (setting it to 0 would
    push every sample the same way). Returns P(LUSC) per sample."""
    genes = model_json["genes"]
    mean = pd.Series(model_json["mean"], index=genes)
    Xg = X.reindex(genes).astype("float64").T.fillna(mean).values
    z = (Xg - mean.values) / np.array(model_json["scale"])
    return 1 / (1 + np.exp(-(model_json["intercept"] + z @ np.array(model_json["coef"]))))
