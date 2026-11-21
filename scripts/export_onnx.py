"""Export a trained category to ONNX (fp32) and a statically quantized INT8 model.

Checks numerical parity against PyTorch on test images and writes
results/export/<category>.json (parity, file sizes) as a DVC metric.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import mlflow
import numpy as np
import torch

from anomaly import tracking
from anomaly.config import PROJECT_ROOT, load_params
from anomaly.data import list_test, list_train
from anomaly.export import export_onnx, quantize_int8
from anomaly.onnx_infer import load_onnx_model
from anomaly.patchcore import iter_batches, load_model, predict

N_PARITY = 16
N_CALIB = 32


def main() -> None:
    """Export, quantize and parity-check one category."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    params = load_params()
    cat, prep = args.category, params.preprocess
    torch.set_num_threads(os.cpu_count() or 1)
    model_dir = PROJECT_ROOT / "models" / cat
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))

    model, _, _ = load_model(model_dir / "patchcore.pt")
    fp32_path = model_dir / "model_fp32.onnx"
    int8_path = model_dir / "model_int8.onnx"
    export_onnx(model, fp32_path, prep.crop)

    # Calibrate activation ranges on good memory-bank images only (no defects, no test data).
    calib_names = set(meta["calib_images"])
    good = [p for p in list_train(params.data_root, cat) if p.name not in calib_names]
    random.Random(params.train.seed).shuffle(good)
    calib_batches = list(iter_batches(good[:N_CALIB], 1, prep))
    quantize_int8(fp32_path, int8_path, calib_batches)

    test_paths = [s.path for s in list_test(params.data_root, cat)]
    random.Random(params.train.seed).shuffle(test_paths)
    parity_paths = sorted(test_paths[:N_PARITY])
    ref_scores, ref_maps = predict(model, iter_batches(parity_paths, 4, prep))

    report: dict[str, object] = {"category": cat, "n_parity_images": len(parity_paths)}
    for variant, path in (("fp32", fp32_path), ("int8", int8_path)):
        onnx_model = load_onnx_model(model_dir, variant)
        s, m = [], []
        for batch in iter_batches(parity_paths, 4, prep):
            bs, bm = onnx_model.predict(batch)
            s.append(bs)
            m.append(bm)
        scores, maps = np.concatenate(s), np.concatenate(m)
        report[variant] = {
            "file_mb": round(path.stat().st_size / 1e6, 1),
            "max_abs_score_diff": float(np.abs(scores - ref_scores).max()),
            "max_rel_score_diff": float((np.abs(scores - ref_scores) / ref_scores).max()),
            "max_abs_map_diff": float(np.abs(maps - ref_maps).max()),
        }
        print(f"{cat} [{variant}] {report[variant]}")

    out = PROJECT_ROOT / "results" / "export" / f"{cat}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    tracking.setup()
    with mlflow.start_run(run_id=meta["mlflow_run_id"]):
        for variant in ("fp32", "int8"):
            for k, v in report[variant].items():  # type: ignore[union-attr]
                mlflow.log_metric(f"export.{variant}.{k}", v)


if __name__ == "__main__":
    main()
