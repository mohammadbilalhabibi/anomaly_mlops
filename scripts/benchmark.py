"""Measure CPU inference latency per image (batch size 1) for each backend.

For one category and every backend (PyTorch, ONNX fp32,
ONNX INT8) it times `runs` single-image predictions after `warmup` runs and
reports median and p95 in milliseconds, separately for preprocessing
(decode + resize + normalize) and model inference.
Writes results/latency/<category>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from anomaly.config import PROJECT_ROOT, load_params
from anomaly.data import list_test
from anomaly.onnx_infer import load_onnx_model
from anomaly.patchcore import load_model
from anomaly.preprocess import open_image, preprocess_image


def cpu_name() -> str:
    """Best-effort CPU model name."""
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def summarize(ms: list[float]) -> dict[str, float]:
    """Median, p95 and mean of a list of millisecond timings."""
    arr = np.asarray(ms)
    return {
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
    }


def time_backend(infer, inputs: list[np.ndarray], warmup: int) -> list[float]:
    """Time `infer` on each input after warm-up; returns milliseconds per call."""
    for x in inputs[:warmup]:
        infer(x)
    times = []
    for x in inputs:
        t0 = time.perf_counter()
        infer(x)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main() -> None:
    """Benchmark one trained category and write results/latency/<category>.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    params = load_params()
    cat, bp, prep = args.category, params.benchmark, params.preprocess
    threads = os.cpu_count() or 1
    torch.set_num_threads(threads)
    model_dir = PROJECT_ROOT / "models" / cat
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))

    paths: list[Path] = [s.path for s in list_test(params.data_root, cat)]
    raw = [paths[i % len(paths)].read_bytes() for i in range(bp.runs)]
    pre_ms: list[float] = []
    inputs = []
    for data in raw:
        t0 = time.perf_counter()
        x = preprocess_image(open_image(data), prep.resize, prep.crop)[None]
        pre_ms.append((time.perf_counter() - t0) * 1000)
        inputs.append(x)

    report: dict[str, object] = {
        "category": cat,
        "cpu": cpu_name(),
        "logical_cores": threads,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "batch_size": 1,
        "runs": bp.runs,
        "warmup": bp.warmup,
        "memory_bank_size": meta["memory_bank_size"],
        "preprocess": summarize(pre_ms),
    }
    torch_model, _, _ = load_model(model_dir / "patchcore.pt")

    def torch_infer(x: np.ndarray, model: torch.nn.Module = torch_model) -> None:
        with torch.inference_mode():
            model(torch.from_numpy(x))

    report["torch"] = summarize(time_backend(torch_infer, inputs, bp.warmup))
    del torch_model
    for variant in ("fp32", "int8"):
        onnx_model = load_onnx_model(model_dir, variant)
        report[f"onnx_{variant}"] = summarize(time_backend(onnx_model.predict, inputs, bp.warmup))
        del onnx_model

    out = PROJECT_ROOT / "results" / "latency" / f"{cat}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
