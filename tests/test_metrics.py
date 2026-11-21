"""Tests for AUROC and AUPRO metrics, including a brute-force check of the PRO curve."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from anomaly.metrics import aupro, image_auroc, pixel_auroc, pro_curve


def test_image_auroc_perfect_and_inverted() -> None:
    labels = np.array([0, 0, 1, 1])
    assert image_auroc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert image_auroc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_pixel_auroc_perfect() -> None:
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[1, 2:4, 2:4] = 1
    assert pixel_auroc(masks, masks.astype(float)) == 1.0


def brute_force_pro(masks: np.ndarray, maps: np.ndarray, t: float) -> tuple[float, float]:
    """Reference PRO and FPR at one threshold, region by region."""
    pred = maps >= t
    overlaps = []
    for i in range(masks.shape[0]):
        labeled, n = ndimage.label(masks[i])
        for r in range(1, n + 1):
            region = labeled == r
            overlaps.append(pred[i][region].mean())
    fpr = (pred & (masks == 0)).sum() / (masks == 0).sum()
    return float(fpr), float(np.mean(overlaps))


def test_pro_curve_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    masks = np.zeros((3, 16, 16), dtype=np.uint8)
    masks[0, 2:6, 2:6] = 1
    masks[0, 10:12, 10:15] = 1  # second, smaller region in the same image
    masks[2, 5:13, 3:7] = 1
    maps = rng.random(masks.shape) + 0.6 * masks
    fpr, pro = pro_curve(masks, maps)
    for t in np.quantile(maps, [0.2, 0.5, 0.8, 0.95]):
        ref_fpr, ref_pro = brute_force_pro(masks, maps, t)
        # Curve point j includes every pixel scoring >= the j-th largest distinct score.
        idx = int((np.unique(maps) >= t).sum())
        np.testing.assert_allclose(fpr[idx], ref_fpr, atol=1e-12)
        np.testing.assert_allclose(pro[idx], ref_pro, atol=1e-12)


def test_aupro_perfect_and_bounds() -> None:
    masks = np.zeros((2, 16, 16), dtype=np.uint8)
    masks[0, 4:8, 4:8] = 1
    assert aupro(masks, masks.astype(float)) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    random_score = aupro(masks, rng.random(masks.shape))
    assert 0.0 <= random_score < 0.5


def test_pro_requires_regions() -> None:
    with pytest.raises(ValueError):
        pro_curve(np.zeros((1, 4, 4)), np.zeros((1, 4, 4)))


def test_quality_gate() -> None:
    from anomaly.metrics import quality_gate

    ref = {"image_auroc": 1.0, "pixel_auroc": 0.98, "aupro": 0.92}
    ok = quality_gate(ref, {"image_auroc": 1.0, "pixel_auroc": 0.975, "aupro": 0.91}, 0.02)
    assert ok["passed"]
    # The real failure this gate exists for: image AUROC intact, localisation destroyed.
    bad = quality_gate(ref, {"image_auroc": 1.0, "pixel_auroc": 0.77, "aupro": 0.33}, 0.02)
    assert not bad["passed"] and bad["drops"]["aupro"] > 0.5
