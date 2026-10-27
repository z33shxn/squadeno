"""Unit tests that need no downloaded data (they run in CI)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from luadlusc import models
from luadlusc.config import AKIYAMA_CLE_GENES, HAMANEH2022_WEIGHTS, SEEDS
from luadlusc.io import barcode_parts, collapse_to_symbols
from luadlusc.metrics import ece, summary


def test_seeds_match_lab_convention():
    assert SEEDS == (42, 123, 456, 789, 1337)


def test_barcode_parts():
    b = barcode_parts("TCGA-55-7726-01A")
    assert b["tss"] == "55" and b["patient"] == "TCGA-55-7726" and b["sample_type"] == "01" and b["vial"] == "A"
    assert barcode_parts("TCGA-18-3421-11A")["sample_type"] == "11"
    with pytest.raises(ValueError):
        barcode_parts("not-a-barcode")


def test_collapse_keeps_highest_row_without_mixing():
    m = pd.DataFrame({"s1": [1.0, 5.0, 2.0], "s2": [1.0, 6.0, 2.0]}, index=["e1", "e2", "e3_PAR_Y"])
    out = collapse_to_symbols(m, pd.Series({"e1": "G", "e2": "G", "e3_PAR_Y": "H"}))
    assert list(out.index) == ["G"]
    assert out.loc["G"].tolist() == [5.0, 6.0]


def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1])
    assert ece(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert ece(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_summary_fields():
    s = summary(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.4, 0.6]))
    assert s["auroc"] == 1.0 and s["accuracy"] == 1.0 and s["n_lusc"] == 2


def test_institution_folds_never_share_an_institution():
    rng = np.random.default_rng(0)
    n = 300
    groups = rng.choice([f"site{i}" for i in range(20)], n)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 5))
    for tr, te in models.folds("institution", X, y, groups, 42):
        assert not set(groups[tr]) & set(groups[te])
        assert len(set(tr) & set(te)) == 0


def test_expression_filter_is_fit_on_training_data_only():
    f = models.ExpressedGenes(threshold=1.0, min_fraction=0.5)
    Xtr = np.array([[2.0, 0.0], [2.0, 0.0]])
    f.fit(Xtr)
    assert f.transform(np.array([[0.0, 9.0]])).shape == (1, 1)


def _synthetic(n=200, p=40, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(p, n)) + 4
    X[0] += 3 * y      # LUSC-high marker
    X[1] += 3 * (1 - y)  # LUAD-high marker
    X[2] += 2 * y
    X[3] += 2 * (1 - y)
    genes = [f"g{i}" for i in range(p)]
    return pd.DataFrame(X, index=genes), y


def test_rank_rules_learn_the_planted_markers():
    X, y = _synthetic()
    rules = models.fit_rank_rules(X, y, k=2, min_median=0.0)
    assert {r[0] for r in rules} <= {"g0", "g2"} and {r[1] for r in rules} <= {"g1", "g3"}
    p, n = models.predict_rank_rules(X, rules)
    assert n == 2 and roc_auc_score(y, p) > 0.95


def test_rank_rules_skip_missing_genes():
    X, y = _synthetic()
    rules = models.fit_rank_rules(X, y, k=2, min_median=0.0)
    _, n = models.predict_rank_rules(X.drop(index=rules[0][0]), rules)
    assert n == 1


def test_baselines_direction():
    genes = list(HAMANEH2022_WEIGHTS) + ["AGR2"]
    rng = np.random.default_rng(1)
    y = np.array([0] * 30 + [1] * 30)
    X = pd.DataFrame(rng.normal(5, 0.3, size=(len(genes), 60)), index=genes)
    for g, w in HAMANEH2022_WEIGHTS.items():
        X.loc[g] += np.where(y == 1, -w, w) * 3   # negative weight = higher in LUSC
    X.loc["AGR2"] = np.where(y == 1, 3.0, 8.0)
    X.loc["KRT5"] = np.where(y == 1, 9.0, 2.0)
    assert roc_auc_score(y, models.hamaneh2022(X)[0]) > 0.95
    assert (models.li2019(X) == y).all()


def test_leakage_canary_detects_selection_before_split():
    """With random labels, selecting features on all samples first inflates AUROC; selecting inside folds does not."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 2000))
    y = rng.integers(0, 2, 120)
    def cv_auc(leaky):
        Xs = SelectKBest(f_classif, k=20).fit_transform(X, y) if leaky else X
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
            Xt = Xs if leaky else SelectKBest(f_classif, k=20).fit(Xs[tr], y[tr]).transform(Xs)
            p[te] = LogisticRegression(max_iter=2000).fit(Xt[tr], y[tr]).predict_proba(Xt[te])[:, 1]
        return roc_auc_score(y, p)
    assert cv_auc(leaky=True) > cv_auc(leaky=False) + 0.15


def test_lipid_module_is_the_published_list():
    assert set(AKIYAMA_CLE_GENES) == {"ABCA12", "ALOXE3", "ALOX12B", "CYP4F22", "CERS3", "ABHD5", "PNPLA1", "ELOVL4"}


def test_json_model_matches_sklearn_pipeline(tmp_path):
    """The exported JSON model must give exactly the same probabilities as the fitted pipeline."""
    import json

    X, y = _synthetic(n=120, p=30, seed=5)
    pipe = models.logistic(0.1).fit(X.T.values, y)
    genes = list(np.array(X.index)[pipe[0].keep_])
    mj = dict(intercept=float(pipe[2].intercept_[0]), genes=genes, mean=list(pipe[1].mean_), scale=list(pipe[1].scale_), coef=list(pipe[2].coef_[0]))
    p_json = models.predict_from_json(json.loads(json.dumps(mj)), X)
    p_sk = pipe.predict_proba(X.T.values)[:, 1]
    assert np.allclose(p_json, p_sk, atol=1e-10)


def test_predict_detects_tpm_and_flags(tmp_path):
    import json

    from luadlusc.predict import score

    X, y = _synthetic(n=60, p=30, seed=7)
    pipe = models.logistic(0.1).fit(X.T.values, y)
    genes = list(np.array(X.index)[pipe[0].keep_])
    mj = dict(intercept=float(pipe[2].intercept_[0]), genes=genes, mean=list(pipe[1].mean_), scale=list(pipe[1].scale_), coef=list(pipe[2].coef_[0]))
    rules = models.fit_rank_rules(X, y, k=2, min_median=0.0)
    (tmp_path / "m.json").write_text(json.dumps(mj))
    (tmp_path / "r.json").write_text(json.dumps([{"lusc_gene": a, "luad_gene": b, "training_score": s} for a, b, s in rules]))
    tpm = np.power(2.0, X) - 1  # give TPM; score() must convert back to log2(TPM+1)
    tpm.iloc[0, 0] = 60.0  # make sure the TPM detector (max > 50) fires
    out = score(tpm, tmp_path / "m.json", tmp_path / "r.json")
    assert list(out.columns) == ["p_lusc_model", "rules_voting_lusc", "call", "check"]
    assert ((out.call == "LUSC").astype(int).values == y).mean() > 0.9
