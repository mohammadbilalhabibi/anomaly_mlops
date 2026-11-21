"""Decision thresholds: label-free rules, decision metrics and label-based references.

The served model only ever uses a label-free rule on held-out good images.
`best_f1_threshold`, `few_label_study` and `recall_at_fpr` use defect labels and
exist only to analyse how much better a threshold could be.
"""

from __future__ import annotations

import numpy as np

LABEL_FREE_RULES = ("p99", "max", "p95", "mean_3std")


def label_free_thresholds(calib_scores: np.ndarray) -> dict[str, float]:
    """Thresholds computed from scores of held-out good images only."""
    s = np.asarray(calib_scores, dtype=np.float64)
    return {
        "p99": float(np.percentile(s, 99)),
        "max": float(s.max()),
        "p95": float(np.percentile(s, 95)),
        "mean_3std": float(s.mean() + 3 * s.std()),
    }


def decision_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall, false-positive rate and F1 when flagging `score > threshold`."""
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) > threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _candidates(scores: np.ndarray) -> np.ndarray:
    """Midpoints between sorted unique scores, plus one value below the minimum."""
    u = np.unique(scores)
    mids = (u[1:] + u[:-1]) / 2
    return np.r_[u[0] - 1e-6, mids]


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold with the highest F1 on the given labelled scores (ties: lowest threshold)."""
    scores = np.asarray(scores, dtype=np.float64)
    best_t, best_f1 = float("nan"), -1.0
    for t in _candidates(scores):
        f1 = decision_metrics(labels, scores, t)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Best recall reachable by any threshold whose false-positive rate is <= `max_fpr`."""
    scores = np.asarray(scores, dtype=np.float64)
    best = 0.0
    for t in _candidates(scores):
        m = decision_metrics(labels, scores, t)
        if m["fpr"] <= max_fpr:
            best = max(best, m["recall"])
    return best


def few_label_study(
    labels: np.ndarray,
    scores: np.ndarray,
    n_good: int = 5,
    n_defect: int = 5,
    n_splits: int = 20,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Pick the best-F1 threshold on a few labelled images, evaluate on the rest.

    Returns mean and std of precision, recall, FPR and F1 over `n_splits` random splits.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    good, defect = np.where(~labels)[0], np.where(labels)[0]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_splits):
        pick = np.r_[
            rng.choice(good, n_good, replace=False), rng.choice(defect, n_defect, replace=False)
        ]
        rest = np.setdiff1d(np.arange(len(labels)), pick)
        t = best_f1_threshold(labels[pick], scores[pick])
        rows.append(decision_metrics(labels[rest], scores[rest], t))
    keys = ("precision", "recall", "fpr", "f1")
    return {
        k: {
            "mean": float(np.mean([r[k] for r in rows])),
            "std": float(np.std([r[k] for r in rows])),
        }
        for k in keys
    }
