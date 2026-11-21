"""Tests for threshold rules, label-based references, profiling parsing and coreset nesting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anomaly.thresholds import (
    best_f1_threshold,
    decision_metrics,
    few_label_study,
    label_free_thresholds,
    recall_at_fpr,
)


def test_label_free_rules() -> None:
    calib = np.array([1.0, 2.0, 3.0, 4.0])
    t = label_free_thresholds(calib)
    assert t["max"] == 4.0
    assert t["p99"] == pytest.approx(np.percentile(calib, 99))
    assert t["p95"] < t["p99"] < t["max"]
    assert t["mean_3std"] == pytest.approx(2.5 + 3 * calib.std())


def test_decision_metrics_counts() -> None:
    labels = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.7, 0.9])
    m = decision_metrics(labels, scores, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(0.5)


def test_best_f1_and_recall_at_fpr_on_separable_scores() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = best_f1_threshold(labels, scores)
    assert 0.3 < t < 0.8
    assert decision_metrics(labels, scores, t)["f1"] == 1.0
    assert recall_at_fpr(labels, scores, 0.0) == 1.0


def test_recall_at_fpr_with_overlap() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.95, 0.5, 0.9])
    # Without false alarms only scores above 0.95 may be flagged: no defect qualifies.
    assert recall_at_fpr(labels, scores, 0.0) == 0.0
    assert recall_at_fpr(labels, scores, 0.25) == 1.0


def test_few_label_study_is_deterministic_and_bounded() -> None:
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(30), np.ones(30)].astype(int)
    scores = np.r_[rng.normal(0, 1, 30), rng.normal(2, 1, 30)]
    a = few_label_study(labels, scores, n_splits=5, seed=1)
    assert a == few_label_study(labels, scores, n_splits=5, seed=1)
    for k in ("precision", "recall", "fpr", "f1"):
        assert 0.0 <= a[k]["mean"] <= 1.0 and a[k]["std"] >= 0.0


def test_parse_profile_splits_stages_and_skips_warmup() -> None:
    from anomaly.profiling import parse_profile

    stages = {"node_conv": "backbone", "node_matmul_1": "nn_search", "node_pad": "postprocess"}
    events = []
    for run, ts in enumerate((0, 100, 200)):
        events.append({"cat": "Session", "name": "model_run", "ts": ts, "dur": 90})
        scale = 1000 if run == 0 else 1  # the warm-up run is slow and must be ignored
        events += [
            {"cat": "Node", "name": "node_Conv_7_token_3_kernel_time", "ts": ts + 1,
             "dur": 10 * scale},
            {"cat": "Node", "name": "node_matmul_1/MatMulScaleFusion/_kernel_time",
             "ts": ts + 2, "dur": 30 * scale},
            {"cat": "Node", "name": "node_pad_kernel_time", "ts": ts + 3, "dur": 10 * scale},
            {"cat": "Node", "name": "node_pad_fence_before", "ts": ts + 3, "dur": 99},
        ]  # fmt: skip
    out = parse_profile(events, stages, warmup=1)
    assert out["profiled_runs"] == 2
    assert out["nn_search_share"] == pytest.approx(0.6)
    assert out["backbone_share"] == pytest.approx(0.2)  # unknown quantized conv -> backbone
    assert out["nn_search_ms"] == pytest.approx(0.03)


def test_node_stages_on_exported_model(models_dir: Path) -> None:
    pytest.importorskip("onnx")
    from anomaly.profiling import node_stages

    stages = node_stages(models_dir / "widget" / "model_fp32.onnx")
    assert set(stages.values()) == {"backbone", "nn_search", "postprocess"}
    nn = [name for name, s in stages.items() if s == "nn_search"]
    assert 3 <= len(nn) <= 12  # distance computation, min and sqrt only


def test_greedy_coreset_is_nested() -> None:
    torch = pytest.importorskip("torch")
    from anomaly.coreset import greedy_coreset

    feats = torch.randn(500, 32, generator=torch.Generator().manual_seed(0))
    big = greedy_coreset(feats, 50, proj_dim=16, seed=0)
    small = greedy_coreset(feats, 10, proj_dim=16, seed=0)
    assert torch.equal(small, big[:10])
