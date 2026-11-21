"""Fit PatchCore for one category on good images only and calibrate its threshold.

Outputs models/<category>/patchcore.pt (memory bank + parameters) and
models/<category>/meta.json (threshold, timings, split, MLflow run id).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import mlflow
import numpy as np
import torch

from anomaly import tracking
from anomaly.config import PROJECT_ROOT, load_params
from anomaly.data import list_train, split_train_calib
from anomaly.patchcore import (
    assemble,
    build_embedder,
    fit_memory_bank,
    iter_batches,
    predict,
    save_model,
)


def main() -> None:
    """Train one category and log the run to MLflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    params = load_params()
    cat, tp, prep = args.category, params.train, params.preprocess
    torch.manual_seed(tp.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    out_dir = PROJECT_ROOT / "models" / cat

    paths = list_train(params.data_root, cat)
    if not paths:
        raise SystemExit(f"no training images for {cat} under {params.data_root}")
    bank_paths, calib_paths = split_train_calib(paths, tp.calib_fraction, tp.seed)

    tracking.setup()
    with mlflow.start_run(run_name=f"{cat}-{int(time.time())}") as run:
        mlflow.set_tags({"category": cat, "git_commit": tracking.git_commit()})
        mlflow.log_params({f"model.{k}": v for k, v in asdict(params.model).items()})
        mlflow.log_params({f"train.{k}": v for k, v in asdict(tp).items()})
        mlflow.log_params({f"preprocess.{k}": v for k, v in asdict(prep).items()})
        mlflow.log_params({"n_bank_images": len(bank_paths), "n_calib_images": len(calib_paths)})

        embedder = build_embedder(params.model, prep)
        t0 = time.perf_counter()
        bank, n_patches = fit_memory_bank(
            embedder,
            iter_batches(bank_paths, tp.batch_size, prep),
            tp.coreset_ratio,
            tp.coreset_proj_dim,
            tp.seed,
        )
        fit_s = time.perf_counter() - t0
        model = assemble(embedder, bank, params.model, prep.crop)

        t0 = time.perf_counter()
        calib_scores, calib_maps = predict(
            model, iter_batches(calib_paths, params.evaluate.batch_size, prep)
        )
        calib_s = time.perf_counter() - t0
        threshold = float(np.percentile(calib_scores, tp.threshold_percentile))
        # Display range for heatmaps: median good pixel -> decision threshold.
        map_vmin = float(np.percentile(calib_maps, 50))

        save_model(out_dir / "patchcore.pt", bank, params.model, prep)
        meta = {
            "category": cat,
            "threshold": threshold,
            "threshold_percentile": tp.threshold_percentile,
            "map_vmin": map_vmin,
            "map_vmax": threshold,
            "calib_score_mean": float(calib_scores.mean()),
            "calib_score_max": float(calib_scores.max()),
            "n_bank_images": len(bank_paths),
            "n_calib_images": len(calib_paths),
            "calib_images": [p.name for p in calib_paths],
            "n_patches": n_patches,
            "memory_bank_size": int(bank.shape[0]),
            "fit_seconds": round(fit_s, 2),
            "calib_seconds": round(calib_s, 2),
            "preprocess": asdict(prep),
            "model": asdict(params.model),
            "mlflow_run_id": run.info.run_id,
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        mlflow.log_metrics(
            {
                "fit_seconds": fit_s,
                "calib_seconds": calib_s,
                "n_patches": n_patches,
                "memory_bank_size": bank.shape[0],
                "threshold": threshold,
            }
        )
        tracking.log_file(out_dir / "meta.json")
        print(
            f"{cat}: bank {bank.shape[0]}/{n_patches} patches, fit {fit_s:.1f}s, "
            f"threshold {threshold:.4f}"
        )


if __name__ == "__main__":
    main()
