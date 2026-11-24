"""Tests for the FastAPI app using a tiny exported ONNX model."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly.config import ServeSettings
from anomaly.serve.api import create_app, index_samples


@pytest.fixture()
def client(models_dir: Path, fake_mvtec: Path):
    settings = ServeSettings(models_dir=models_dir, samples_dir=fake_mvtec, samples_per_category=4)
    with TestClient(create_app(settings)) as c:
        yield c


def _png(path: Path) -> bytes:
    return path.read_bytes()


def test_health_and_categories(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["categories"] == ["widget"]
    assert client.get("/api/categories").json() == [{"category": "widget", "threshold": 1.0}]


def test_samples_listed_and_served(client: TestClient) -> None:
    samples = client.get("/api/samples").json()
    assert len(samples) == 4
    assert {s["defect_type"] for s in samples} == {"good", "scratch"}
    r = client.get(samples[0]["url"])
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert client.get("/api/samples/../../etc").status_code == 404


def test_predict_returns_score_and_heatmaps(client: TestClient, fake_mvtec: Path) -> None:
    img = fake_mvtec / "widget/test/scratch/000.png"
    r = client.post(
        "/api/predict", data={"category": "widget"}, files={"file": ("a.png", _png(img))}
    )
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["score"], float) and isinstance(body["is_anomaly"], bool)
    assert body["is_anomaly"] == (body["score"] > body["threshold"])
    for key in ("image_png", "overlay_png", "heatmap_png"):
        decoded = Image.open(io.BytesIO(base64.b64decode(body[key])))
        assert decoded.size == (64, 64)
    assert body["inference_ms"] > 0 and body["total_ms"] >= body["inference_ms"]


def test_predict_errors(client: TestClient, fake_mvtec: Path) -> None:
    img = _png(fake_mvtec / "widget/test/good/000.png")
    r = client.post("/api/predict", data={"category": "nope"}, files={"file": ("a.png", img)})
    assert r.status_code == 404
    r = client.post(
        "/api/predict", data={"category": "widget"}, files={"file": ("a.png", b"not an image")}
    )
    assert r.status_code == 400


def test_dashboard_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and "Anomaly Inspector" in r.text


def test_app_starts_without_models(tmp_path: Path) -> None:
    settings = ServeSettings(models_dir=tmp_path / "none", samples_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/health").json()["categories"] == []


def test_index_samples_cycles_defect_types(fake_mvtec: Path) -> None:
    found = index_samples(fake_mvtec, ["widget"], 3)
    assert [s.defect_type for s in found.values()] == ["good", "scratch", "good"]


def test_assets_downloaded_from_hf_repo_at_startup(
    models_dir: Path, fake_mvtec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_fetch(repo_id: str, variant: str, dest: Path) -> tuple[Path, Path]:
        calls.append((repo_id, variant, dest))
        return models_dir, fake_mvtec

    monkeypatch.setattr("anomaly.serve.api.fetch_assets", fake_fetch)
    settings = ServeSettings(
        models_dir=tmp_path / "missing",
        samples_dir=tmp_path / "missing",
        hf_assets_repo="someone/assets",
        assets_dir=tmp_path / "assets",
    )
    with TestClient(create_app(settings)) as c:
        assert c.get("/api/health").json()["categories"] == ["widget"]
        assert len(c.get("/api/samples").json()) == 6
    assert calls == [("someone/assets", "int8", tmp_path / "assets")]


def test_fetch_assets_requests_only_one_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    from anomaly.serve.assets import fetch_assets

    seen = {}
    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.snapshot_download = lambda **kw: seen.update(kw)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    models, samples = fetch_assets("someone/assets", "int8", tmp_path)
    assert (models, samples) == (tmp_path / "models", tmp_path / "samples")
    assert seen["repo_id"] == "someone/assets" and seen["local_dir"] == tmp_path
    assert "models/*/model_int8.onnx" in seen["allow_patterns"]
    assert not any("fp32" in p for p in seen["allow_patterns"])


def test_stage_assets_uses_repo_layout(models_dir: Path, fake_mvtec: Path, tmp_path: Path) -> None:
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "upload_assets_hf", root / "scripts" / "upload_assets_hf.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    counts = module.stage_assets(models_dir, fake_mvtec, "int8", 4, tmp_path)
    assert counts == {"categories": 1, "samples": 4}
    assert (tmp_path / "models/widget/model_int8.onnx").is_file()
    assert (tmp_path / "models/widget/meta.json").is_file()
    assert not (tmp_path / "models/widget/model_fp32.onnx").exists()
    assert len(list((tmp_path / "samples/widget/test").rglob("*.png"))) == 4
    assert "cc-by-nc-sa-4.0" in (tmp_path / "README.md").read_text(encoding="utf-8")
