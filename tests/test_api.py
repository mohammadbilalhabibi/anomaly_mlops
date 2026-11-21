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
