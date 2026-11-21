"""ONNX Runtime inference without torch (used by the API, the benchmark and ONNX evaluation)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

VARIANT_FILES = {"fp32": "model_fp32.onnx", "int8": "model_int8.onnx"}


def make_session(path: Path, threads: int = 0) -> ort.InferenceSession:
    """Create a CPU inference session with full graph optimisation."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads > 0:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


@dataclass
class OnnxAnomalyModel:
    """A loaded ONNX PatchCore model plus its calibration metadata."""

    category: str
    variant: str
    session: ort.InferenceSession
    threshold: float
    map_vmin: float
    map_vmax: float
    resize: int
    crop: int

    def predict(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score a preprocessed (b, 3, crop, crop) float32 batch -> (scores, maps)."""
        scores, maps = self.session.run(None, {"image": batch.astype(np.float32, copy=False)})
        return scores, maps


def load_onnx_model(
    category_dir: Path, variant: str = "fp32", threads: int = 0
) -> OnnxAnomalyModel:
    """Load `model_<variant>.onnx` and `meta.json` from a category's model folder."""
    if variant not in VARIANT_FILES:
        raise ValueError(f"variant must be one of {sorted(VARIANT_FILES)}")
    meta = json.loads((category_dir / "meta.json").read_text(encoding="utf-8"))
    return OnnxAnomalyModel(
        category=category_dir.name,
        variant=variant,
        session=make_session(category_dir / VARIANT_FILES[variant], threads),
        threshold=float(meta["threshold"]),
        map_vmin=float(meta["map_vmin"]),
        map_vmax=float(meta["map_vmax"]),
        resize=int(meta["preprocess"]["resize"]),
        crop=int(meta["preprocess"]["crop"]),
    )


def discover_models(
    models_dir: Path, variant: str = "fp32", threads: int = 0
) -> dict[str, OnnxAnomalyModel]:
    """Load every category folder in `models_dir` that has the requested ONNX file."""
    if not models_dir.is_dir():
        return {}
    found: dict[str, OnnxAnomalyModel] = {}
    for d in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if (d / VARIANT_FILES.get(variant, "")).is_file() and (d / "meta.json").is_file():
            found[d.name] = load_onnx_model(d, variant, threads)
    return found
