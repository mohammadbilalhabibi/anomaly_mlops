"""Evaluate a trained category on the MVTec AD test set with one or more backends.

Backends: torch (PyTorch model), fp32 (ONNX), int8 (quantized ONNX).
Writes results/<backend>/<category>.json and logs metrics to the training run in MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from PIL import Image

from anomaly import tracking
from anomaly.config import PROJECT_ROOT, load_params
from anomaly.data import list_test
from anomaly.heatmap import normalize_map, overlay
from anomaly.metrics import aupro, image_auroc, pixel_auroc, quality_gate
from anomaly.onnx_infer import load_onnx_model
from anomaly.patchcore import iter_batches, load_model, predict
from anomaly.preprocess import display_image, load_mask, open_image

BACKENDS = ("torch", "fp32", "int8")


def run_backend(
    backend: str, model_dir: Path, paths: list[Path], batch_size: int, prep
) -> tuple[np.ndarray, np.ndarray]:
    """Score all test images with the given backend."""
    if backend == "torch":
        model, _, _ = load_model(model_dir / "patchcore.pt")
        return predict(model, iter_batches(paths, batch_size, prep))
    onnx_model = load_onnx_model(model_dir, backend)
    scores, maps = [], []
    for batch in iter_batches(paths, batch_size, prep):
        s, m = onnx_model.predict(batch)
        scores.append(s)
        maps.append(m)
    return np.concatenate(scores), np.concatenate(maps)


def save_examples(samples, scores, maps, meta: dict, out_dir: Path, prep) -> list[Path]:
    """Save heatmap overlays for the highest-scoring defect and good test images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.array([s.label for s in samples])
    picks = []
    for label in (1, 0):
        idx = np.where(labels == label)[0]
        picks += list(idx[np.argsort(-scores[idx])][:2])
    written = []
    for i in picks:
        rgb = display_image(open_image(samples[i].path), prep.resize, prep.crop)
        norm = normalize_map(maps[i], meta["map_vmin"], meta["map_vmax"])
        path = out_dir / f"{samples[i].defect_type}_{samples[i].path.stem}.png"
        Image.fromarray(np.concatenate([rgb, overlay(rgb, norm)], axis=1)).save(path)
        written.append(path)
    return written


def main() -> None:
    """Evaluate one category and write measured metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS)
    args = parser.parse_args()

    params = load_params()
    cat, prep, ep = args.category, params.preprocess, params.evaluate
    torch.set_num_threads(os.cpu_count() or 1)
    model_dir = PROJECT_ROOT / "models" / cat
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))

    samples = list_test(params.data_root, cat)
    paths = [s.path for s in samples]
    labels = np.array([s.label for s in samples])
    masks = np.stack([load_mask(s.mask_path, prep.resize, prep.crop) for s in samples])

    results: dict[str, dict] = {}
    tracking.setup()
    with mlflow.start_run(run_id=meta["mlflow_run_id"]):
        for backend in args.backends:
            t0 = time.perf_counter()
            scores, maps = run_backend(backend, model_dir, paths, ep.batch_size, prep)
            elapsed = time.perf_counter() - t0
            pred = scores > meta["threshold"]
            tp = int((pred & (labels == 1)).sum())
            fp = int((pred & (labels == 0)).sum())
            fn = int((~pred & (labels == 1)).sum())
            result = {
                "category": cat,
                "backend": backend,
                "n_test": len(samples),
                "n_defect": int(labels.sum()),
                "image_auroc": image_auroc(labels, scores),
                "pixel_auroc": pixel_auroc(masks, maps),
                "aupro": aupro(masks, maps, ep.pro_fpr_limit),
                "pro_fpr_limit": ep.pro_fpr_limit,
                "threshold": meta["threshold"],
                "precision_at_threshold": tp / (tp + fp) if tp + fp else 0.0,
                "recall_at_threshold": tp / (tp + fn) if tp + fn else 0.0,
                "false_positive_rate_at_threshold": fp / max(int((labels == 0).sum()), 1),
                "eval_seconds": round(elapsed, 2),
                "pixel_metric_resolution": prep.crop,
            }
            if backend != "torch" and "torch" in results:
                result["quality_gate"] = quality_gate(results["torch"], result, ep.max_metric_drop)
                mlflow.log_metric(f"{backend}.gate_passed", float(result["quality_gate"]["passed"]))
            results[backend] = result
            out = PROJECT_ROOT / "results" / backend / f"{cat}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2), encoding="utf-8")
            mlflow.log_metrics(
                {
                    f"{backend}.{k}": v
                    for k, v in result.items()
                    if isinstance(v, float) and k != "pro_fpr_limit"
                }
            )
            print(
                f"{cat} [{backend}] image AUROC {result['image_auroc']:.4f}  "
                f"pixel AUROC {result['pixel_auroc']:.4f}  AUPRO {result['aupro']:.4f}  "
                f"({elapsed:.1f}s)"
            )
            if backend == "torch":
                for p in save_examples(samples, scores, maps, meta, model_dir / "examples", prep):
                    tracking.log_file(p, "examples")

    failed = [b for b, r in results.items() if not r.get("quality_gate", {}).get("passed", True)]
    if failed:
        for b in failed:
            print(
                f"QUALITY GATE FAILED: {cat} [{b}] drops {results[b]['quality_gate']['drops']}",
                file=sys.stderr,
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
