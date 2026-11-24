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


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_replace_blocks_only_touches_marked_sections() -> None:
    m = _script("make_results_table")
    text = "intro\n<!-- A:START -->\nold\n<!-- A:END -->\nmiddle\n<!-- B:START -->x<!-- B:END -->"
    out = m.replace_blocks(text, {"A": "new\n", "C": "unused\n"})
    assert out.startswith("intro\n<!-- A:START -->\nnew\n<!-- A:END -->\nmiddle")
    assert "old" not in out and "<!-- B:START -->x<!-- B:END -->" in out


def test_demo_gif_merges_identical_frames(tmp_path: Path) -> None:
    from PIL import Image

    m = _script("record_demo")
    red, blue = Image.new("RGB", (40, 30), "red"), Image.new("RGB", (40, 30), "blue")
    frames = [(red, 0.0), (red.copy(), 0.1), (blue, 0.2), (blue.copy(), 0.3)]
    size = m.to_gif(frames, tmp_path / "demo.gif", width=20)
    gif = Image.open(tmp_path / "demo.gif")
    assert size > 0 and gif.size == (20, 15) and gif.n_frames == 2


def test_deploy_block_from_measured_json(report_module, tmp_path: Path) -> None:
    assert "Not measured yet" in report_module.deploy_block()
    measured = {
        "category": "bottle", "variant": "int8", "runs": 50, "median_ms": 363.0, "p95_ms": 377.4,
        "server_median_ms": 193.2, "server_p95_ms": 206.4, "inference_median_ms": 128.4,
    }  # fmt: skip
    (tmp_path / "latency_azure.json").write_text(json.dumps(measured))
    block = report_module.deploy_block()
    assert "| Server total: decode, model, heatmap PNGs | 193 ms | 206 ms |" in block
    assert "scaled to zero" not in block
    (tmp_path / "latency_azure.json").write_text(json.dumps({**measured, "cold_start_s": 41.6}))
    assert "**42 s**" in report_module.deploy_block()
