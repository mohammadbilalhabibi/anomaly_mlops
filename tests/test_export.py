"""Tests for ONNX export parity and INT8 quantization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anomaly.data import list_test  # noqa: E402
from anomaly.onnx_infer import discover_models, load_onnx_model  # noqa: E402
from anomaly.patchcore import iter_batches, predict  # noqa: E402

from .conftest import PREP  # noqa: E402


def _batch(fake_mvtec: Path) -> np.ndarray:
    paths = [s.path for s in list_test(fake_mvtec, "widget")]
    return next(iter_batches(paths, 3, PREP))


def test_onnx_fp32_matches_pytorch(models_dir: Path, tiny_model, fake_mvtec: Path) -> None:
    batch = _batch(fake_mvtec)
    ref_scores, ref_maps = predict(tiny_model, [batch])
    model = load_onnx_model(models_dir / "widget", "fp32")
    scores, maps = model.predict(batch)  # batch of 3 also checks the dynamic batch axis
    np.testing.assert_allclose(scores, ref_scores, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(maps, ref_maps, rtol=1e-4, atol=1e-4)


def test_onnx_int8_runs_and_stays_close(models_dir: Path, tiny_model, fake_mvtec: Path) -> None:
    batch = _batch(fake_mvtec)
    ref_scores, _ = predict(tiny_model, [batch])
    model = load_onnx_model(models_dir / "widget", "int8")
    scores, maps = model.predict(batch)
    assert scores.shape == ref_scores.shape and maps.shape == (3, PREP.crop, PREP.crop)
    assert np.all(np.isfinite(scores))
    # Loose bound: random calibration data and a random backbone, so only a sanity check.
    assert np.max(np.abs(scores - ref_scores) / ref_scores) < 0.5


def test_discover_models(models_dir: Path, tmp_path: Path) -> None:
    assert set(discover_models(models_dir, "fp32")) == {"widget"}
    assert discover_models(tmp_path / "missing") == {}
    with pytest.raises(ValueError):
        load_onnx_model(models_dir / "widget", "fp16")


def test_int8_keeps_scorer_in_fp32(models_dir: Path) -> None:
    onnx = pytest.importorskip("onnx")
    graph = onnx.load(str(models_dir / "widget" / "model_int8.onnx")).graph
    quantized_inputs = {n.input[0] for n in graph.node if n.op_type == "QuantizeLinear"}
    blur = [n for n in graph.node if n.op_type == "Conv" and n.input[1].startswith("scorer.")]
    assert {n.input[1] for n in blur} == {"scorer.kernel_h", "scorer.kernel_v"}
    assert not any("DequantizeLinear" in n.input[0] for n in blur), "blur input must stay fp32"
    backbone_convs = [n for n in graph.node if n.op_type == "Conv" and "backbone" in n.input[1]]
    assert backbone_convs and all("DequantizeLinear" in n.input[1] for n in backbone_convs)
    assert not any(name.startswith("scorer.") for name in quantized_inputs)
