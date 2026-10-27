"""Performance metrics. Probabilities are P(LUSC); labels are 1 = LUSC, 0 = LUAD."""

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def summary(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p > threshold).astype(int)
    return dict(n=int(len(y)), n_lusc=int(y.sum()), auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                accuracy=float((pred == y).mean()), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                lusc_recall=float((pred[y == 1] == 1).mean()), luad_recall=float((pred[y == 0] == 0).mean()),
                ece=ece(y, p), brier=float(brier_score_loss(y, p)))


def bootstrap_ci(y, p, fn=roc_auc_score, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
