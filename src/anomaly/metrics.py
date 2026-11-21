"""Anomaly detection metrics: image AUROC, pixel AUROC and AUPRO."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

GATED_METRICS = ("image_auroc", "pixel_auroc", "aupro")


def quality_gate(
    reference: dict[str, float], candidate: dict[str, float], max_drop: float
) -> dict[str, object]:
    """Compare a deployable model against the reference on every gated metric.

    Fails when any metric drops by more than `max_drop` (absolute, e.g. 0.02 = 2 points).
    """
    drops = {k: float(reference[k] - candidate[k]) for k in GATED_METRICS}
    return {
        "passed": all(d <= max_drop for d in drops.values()),
        "max_drop": max_drop,
        "drops": drops,
    }


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Image-level AUROC (labels: 1 = defect)."""
    return float(roc_auc_score(np.asarray(labels).ravel(), np.asarray(scores).ravel()))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """Pixel-level AUROC over all pixels of all test images."""
    return float(roc_auc_score(np.asarray(masks).ravel() > 0, np.asarray(maps).ravel()))


def pro_curve(masks: np.ndarray, maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact per-region-overlap curve: (FPR, PRO) at every distinct threshold.

    PRO at a threshold is the mean, over all connected ground-truth regions, of
    the fraction of that region predicted anomalous. Sorting all pixels once by
    score gives the whole curve in O(n log n): each defect pixel contributes
    1 / (size of its region) / (number of regions) once it is above threshold.
    """
    masks = np.asarray(masks) > 0
    maps = np.asarray(maps, dtype=np.float64)
    if masks.shape != maps.shape:
        raise ValueError("masks and maps must have the same shape")

    weights = np.zeros(masks.shape, dtype=np.float64)
    n_regions = 0
    for i in range(masks.shape[0]):
        labeled, count = ndimage.label(masks[i])
        if count == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        region_w = np.zeros_like(sizes, dtype=np.float64)
        region_w[1:] = 1.0 / sizes[1:]
        weights[i] = region_w[labeled]
        n_regions += count
    n_normal = int((~masks).sum())
    if n_regions == 0 or n_normal == 0:
        raise ValueError("need at least one defect region and one normal pixel")

    order = np.argsort(-maps.ravel(), kind="stable")
    sorted_scores = maps.ravel()[order]
    fp = np.cumsum(~masks.ravel()[order]) / n_normal
    pro = np.cumsum(weights.ravel()[order]) / n_regions
    # Keep only the last index of each run of tied scores (one point per threshold).
    last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    return np.r_[0.0, fp[last_of_tie]], np.r_[0.0, pro[last_of_tie]]


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3) -> float:
    """Area under the PRO curve up to `fpr_limit`, normalized to [0, 1]."""
    fpr, pro = pro_curve(masks, maps)
    cut = np.searchsorted(fpr, fpr_limit, side="right")
    x, y = fpr[:cut], pro[:cut]
    if cut < len(fpr) and x[-1] < fpr_limit:
        # Linearly interpolate the PRO value exactly at the FPR limit.
        x0, x1, y0, y1 = fpr[cut - 1], fpr[cut], pro[cut - 1], pro[cut]
        y_lim = y0 + (y1 - y0) * (fpr_limit - x0) / (x1 - x0)
        x, y = np.r_[x, fpr_limit], np.r_[y, y_lim]
    return float(np.trapezoid(y, x) / fpr_limit)
