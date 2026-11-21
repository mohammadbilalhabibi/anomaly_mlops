"""Tests for heatmap rendering and the generated results tables."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from anomaly.heatmap import LUT, colorize, normalize_map, overlay, to_png_base64

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_and_colorize() -> None:
    norm = normalize_map(np.array([[0.0, 1.0], [2.0, 5.0]]), vmin=1.0, vmax=3.0)
    np.testing.assert_allclose(norm, [[0.0, 0.0], [0.5, 1.0]])
    rgb = colorize(norm)
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], LUT[0])
    np.testing.assert_array_equal(rgb[1, 1], LUT[255])


def test_overlay_keeps_normal_pixels_unchanged() -> None:
    rgb = np.full((4, 4, 3), 100, dtype=np.uint8)
    norm = np.zeros((4, 4))
    norm[0, 0] = 1.0
    out = overlay(rgb, norm)
    np.testing.assert_array_equal(out[1:, 1:], rgb[1:, 1:])
    assert not np.array_equal(out[0, 0], rgb[0, 0])


def test_png_base64_roundtrip() -> None:
    s = to_png_base64(np.zeros((3, 3, 3), dtype=np.uint8))
    assert base64.b64decode(s)[:4] == b"\x89PNG"


@pytest.fixture()
def report_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location(
        "make_results_table", ROOT / "scripts" / "make_results_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    monkeypatch.setattr(module, "RESULTS", tmp_path)
    return module


def test_results_table_uses_measured_json(report_module, tmp_path: Path) -> None:
    assert "No results yet" in report_module.accuracy_table()
    (tmp_path / "torch").mkdir()
    result = {
        "image_auroc": 0.9876, "pixel_auroc": 0.9512, "aupro": 0.9034, "n_test": 10,
        "n_defect": 6, "precision_at_threshold": 1.0, "recall_at_threshold": 0.5,
        "false_positive_rate_at_threshold": 0.0,
    }  # fmt: skip
    (tmp_path / "torch" / "bottle.json").write_text(json.dumps(result))
    table = report_module.accuracy_table()
    assert "| bottle | 98.8 | 95.1 | 90.3 |" in table
    assert "| bottle | 10 (6) | 100.0 | 50.0 | 0.0 |" in report_module.threshold_table()
