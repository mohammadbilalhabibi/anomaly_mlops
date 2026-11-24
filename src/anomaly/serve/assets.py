"""Download models and dashboard samples from a private Hugging Face repo at startup.

Used by the deployed container: the Docker image contains no MVTec-derived files (CC BY-NC-SA
4.0), so it can be public. The models and sample images live in a private model repo and are
fetched once when the app starts, with a read-only token from the `HF_TOKEN` environment
variable (read by huggingface_hub itself, never logged).

Repo layout (written by scripts/upload_assets_hf.py):
    models/<category>/model_<variant>.onnx
    models/<category>/meta.json
    samples/<category>/test/<defect_type>/<file>.png
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

# uvicorn's logger, so the download time shows up in the container logs.
log = logging.getLogger("uvicorn.error")


def asset_patterns(variant: str) -> list[str]:
    """Files to download: one ONNX variant plus metadata per category, and the samples."""
    return [f"models/*/model_{variant}.onnx", "models/*/meta.json", "samples/**"]


def fetch_assets(repo_id: str, variant: str, dest: Path) -> tuple[Path, Path]:
    """Download the assets of `repo_id` into `dest`; returns (models_dir, samples_dir)."""
    from huggingface_hub import snapshot_download  # optional dependency: `pip install .[deploy]`

    start = time.perf_counter()
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=dest,
        allow_patterns=asset_patterns(variant),
    )
    log.info("downloaded assets from %s in %.1f s", repo_id, time.perf_counter() - start)
    return dest / "models", dest / "samples"
