"""FastAPI app: ONNX anomaly scoring API plus a static dashboard.

Models are loaded once at startup (lifespan). No torch is imported here.
Run: uvicorn anomaly.serve.api:app --port 8000
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from anomaly.config import ServeSettings
from anomaly.data import IMAGE_SUFFIXES
from anomaly.heatmap import colorize, normalize_map, overlay, to_png_base64
from anomaly.onnx_infer import OnnxAnomalyModel, discover_models
from anomaly.preprocess import display_image, open_image, preprocess_image

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class Sample:
    """A test image offered in the dashboard."""

    id: str
    category: str
    defect_type: str
    path: Path


def index_samples(samples_dir: Path, categories: list[str], per_category: int) -> dict[str, Sample]:
    """Pick a few test images per category, cycling over defect types (good first)."""
    found: dict[str, Sample] = {}
    for cat in categories:
        test_dir = samples_dir / cat / "test"
        if not test_dir.is_dir():
            continue
        types = sorted(
            (d for d in test_dir.iterdir() if d.is_dir()), key=lambda d: (d.name != "good", d.name)
        )
        files = {
            d.name: sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            for d in types
        }
        picked, rank = 0, 0
        while picked < per_category and any(rank < len(v) for v in files.values()):
            for defect, paths in files.items():
                if rank < len(paths) and picked < per_category:
                    sid = f"{cat}-{defect}-{paths[rank].stem}"
                    found[sid] = Sample(sid, cat, defect, paths[rank])
                    picked += 1
            rank += 1
    return found


def score_image(model: OnnxAnomalyModel, data: bytes) -> dict[str, object]:
    """Preprocess, score and render one image; returns the JSON response body."""
    t_start = time.perf_counter()
    img = open_image(data)
    x = preprocess_image(img, model.resize, model.crop)[None]
    t_infer = time.perf_counter()
    scores, maps = model.predict(x)
    infer_ms = (time.perf_counter() - t_infer) * 1000
    score = float(scores[0])
    norm = normalize_map(maps[0], model.map_vmin, model.map_vmax)
    rgb = display_image(img, model.resize, model.crop)
    body = {
        "category": model.category,
        "variant": model.variant,
        "score": score,
        "threshold": model.threshold,
        "score_ratio": score / model.threshold,
        "is_anomaly": score > model.threshold,
        "inference_ms": round(infer_ms, 2),
        "image_png": to_png_base64(rgb),
        "overlay_png": to_png_base64(overlay(rgb, norm)),
        "heatmap_png": to_png_base64(colorize(norm)),
    }
    body["total_ms"] = round((time.perf_counter() - t_start) * 1000, 2)
    return body


def create_app(settings: ServeSettings | None = None) -> FastAPI:
    """Build the FastAPI app; models and the sample index load once in the lifespan."""
    settings = settings or ServeSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.models = discover_models(
            settings.models_dir, settings.model_variant, settings.ort_threads
        )
        app.state.samples = index_samples(
            settings.samples_dir, list(app.state.models), settings.samples_per_category
        )
        yield

    app = FastAPI(title="anomaly_mlops", version="0.1.0", lifespan=lifespan)

    def get_model(category: str) -> OnnxAnomalyModel:
        model = app.state.models.get(category)
        if model is None:
            raise HTTPException(404, f"no model for category '{category}'")
        return model

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Liveness plus which models are loaded."""
        return {
            "status": "ok",
            "variant": settings.model_variant,
            "categories": sorted(app.state.models),
        }

    @app.get("/api/categories")
    def categories() -> list[dict[str, object]]:
        """Loaded categories with their calibrated thresholds."""
        return [{"category": c, "threshold": m.threshold} for c, m in app.state.models.items()]

    @app.get("/api/samples")
    def samples() -> list[dict[str, str]]:
        """Sample test images available on this server."""
        return [
            {
                "id": s.id,
                "category": s.category,
                "defect_type": s.defect_type,
                "url": f"/api/samples/{s.id}",
            }
            for s in app.state.samples.values()
        ]

    @app.get("/api/samples/{sample_id}")
    def sample_image(sample_id: str) -> FileResponse:
        """Serve one sample image by id (ids come from the startup index, not from paths)."""
        sample = app.state.samples.get(sample_id)
        if sample is None:
            raise HTTPException(404, "unknown sample")
        return FileResponse(sample.path)

    @app.post("/api/predict")
    def predict(
        category: Annotated[str, Form()], file: Annotated[UploadFile, File()]
    ) -> dict[str, object]:
        """Score an uploaded image: anomaly score, decision and heatmap images (base64 PNG)."""
        model = get_model(category)
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "image larger than 10 MB")
        try:
            return score_image(model, data)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(400, "could not read the image") from exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
