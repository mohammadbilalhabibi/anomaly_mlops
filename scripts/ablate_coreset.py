"""Coreset ablation: how much speed does a smaller memory bank buy, and what does it cost?

For each category and coreset ratio this repeats the pipeline steps — train
(memory bank + recalibrated threshold), export (ONNX fp32 -> backbone-only INT8),
evaluate and benchmark — with the same seed, calibration split and benchmark
settings as the main pipeline. The existing ratio-0.1 models are not retrained;
their INT8 models are re-benchmarked and profiled in the same session so all
ratios are timed under the same conditions.

Run:  python scripts/ablate_coreset.py
Writes results/ablation_coreset/<category>_<ratio>.json, models to
models/ablation_coreset/ (gitignored) and runs to the MLflow experiment
"coreset_ablation". The main pipeline (params.yaml, dvc.lock, results/*) is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import torch

from anomaly import tracking
from anomaly.bench import cpu_name, summarize, time_calls
from anomaly.config import PROJECT_ROOT, Params, load_params
from anomaly.coreset import greedy_coreset
from anomaly.data import list_test, list_train, split_train_calib
from anomaly.export import export_onnx, quantize_int8
from anomaly.metrics import aupro, image_auroc, pixel_auroc
from anomaly.model import num_coreset
from anomaly.onnx_infer import load_onnx_model
from anomaly.patchcore import assemble, build_embedder, extract_patches, iter_batches, predict
from anomaly.preprocess import load_mask, open_image, preprocess_image
from anomaly.profiling import profile_stages
from anomaly.thresholds import decision_metrics

EXPERIMENT = "coreset_ablation"
DEFAULT_RATIO = 0.1
OUT_DIR = PROJECT_ROOT / "results" / "ablation_coreset"
MODEL_DIR = PROJECT_ROOT / "models" / "ablation_coreset"
N_QUANT_CALIB = 32  # same as scripts/export_onnx.py
PROFILE_WARMUP = 3


class TestSet:
    """Test images, labels, masks and the fixed benchmark inputs of one category."""

    def __init__(self, params: Params, category: str) -> None:
        prep = params.preprocess
        samples = list_test(params.data_root, category)
        self.paths = [s.path for s in samples]
        self.labels = np.array([s.label for s in samples])
        self.masks = np.stack([load_mask(s.mask_path, prep.resize, prep.crop) for s in samples])
        runs = params.benchmark.runs  # same inputs as scripts/benchmark.py
        raw = [self.paths[i % len(self.paths)].read_bytes() for i in range(runs)]
        self.bench_inputs = [
            preprocess_image(open_image(b), prep.resize, prep.crop)[None] for b in raw
        ]


def evaluate_int8(model_dir: Path, test: TestSet, params: Params, threshold: float) -> dict:
    """Image/pixel AUROC, AUPRO and decisions of an INT8 model on the test set."""
    onnx_model = load_onnx_model(model_dir, "int8")
    scores, maps = [], []
    for batch in iter_batches(test.paths, params.evaluate.batch_size, params.preprocess):
        s, m = onnx_model.predict(batch)
        scores.append(s)
        maps.append(m)
    s, m = np.concatenate(scores), np.concatenate(maps)
    d = decision_metrics(test.labels, s, threshold)
    return {
        "image_auroc": image_auroc(test.labels, s),
        "pixel_auroc": pixel_auroc(test.masks, m),
        "aupro": aupro(test.masks, m, params.evaluate.pro_fpr_limit),
        "precision_at_threshold": d["precision"],
        "recall_at_threshold": d["recall"],
        "false_positive_rate_at_threshold": d["fpr"],
    }


def speed(model_dir: Path, test: TestSet, params: Params, tag: str) -> dict:
    """INT8 latency (same settings as the main benchmark) and the per-stage time split."""
    onnx_model = load_onnx_model(model_dir, "int8")
    latency = summarize(time_calls(onnx_model.predict, test.bench_inputs, params.benchmark.warmup))
    del onnx_model
    prefix = Path(os.environ.get("TEMP", PROJECT_ROOT)) / f"ort_profile_{tag}"
    stages = profile_stages(
        model_dir / "model_int8.onnx", test.bench_inputs[:20], PROFILE_WARMUP, prefix
    )
    return {"int8_latency": latency, "profile": stages}


def default_row(params: Params, category: str, test: TestSet) -> dict:
    """Ratio 0.1: metrics from the main pipeline results, speed re-measured in this session."""
    model_dir = PROJECT_ROOT / "models" / category
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    int8 = json.loads((PROJECT_ROOT / "results" / "int8" / f"{category}.json").read_text())
    keys = (
        "image_auroc", "pixel_auroc", "aupro", "precision_at_threshold",
        "recall_at_threshold", "false_positive_rate_at_threshold",
    )  # fmt: skip
    return {
        "source": "main pipeline (models/<category>, results/int8), not retrained",
        "memory_bank_size": meta["memory_bank_size"],
        "threshold": meta["threshold"],
        "int8": {k: int8[k] for k in keys},
        "int8_file_mb": round((model_dir / "model_int8.onnx").stat().st_size / 1e6, 1),
        **speed(model_dir, test, params, f"{category}_0.1"),
    }


def ablation_row(
    params: Params,
    category: str,
    ratio: float,
    patches: torch.Tensor,
    embedder,
    calib_paths: list[Path],
    quant_paths: list[Path],
    test: TestSet,
    default_bank: torch.Tensor,
) -> dict:
    """Train, export, evaluate and benchmark one (category, ratio) pair."""
    tp, prep, mp = params.train, params.preprocess, params.model
    out = MODEL_DIR / f"{category}_{ratio}"
    out.mkdir(parents=True, exist_ok=True)

    n_select = num_coreset(patches.shape[0], ratio)
    t0 = time.perf_counter()
    idx = greedy_coreset(patches, n_select, proj_dim=tp.coreset_proj_dim, seed=tp.seed)
    coreset_s = time.perf_counter() - t0
    bank = patches[idx].clone()
    # Greedy k-center selection is nested: with the same seed, the first n picks of the
    # 10% run are exactly the n picks of a smaller run. Record whether that held here.
    prefix_diff = float((bank - default_bank[:n_select]).abs().max())

    model = assemble(embedder, bank, mp, prep.crop)
    calib_scores, calib_maps = predict(
        model, iter_batches(calib_paths, params.evaluate.batch_size, prep)
    )
    threshold = float(np.percentile(calib_scores, tp.threshold_percentile))
    meta = {
        "category": category,
        "coreset_ratio": ratio,
        "threshold": threshold,
        "map_vmin": float(np.percentile(calib_maps, 50)),
        "map_vmax": threshold,
        "memory_bank_size": int(bank.shape[0]),
        "preprocess": asdict(prep),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fp32_path, int8_path = out / "model_fp32.onnx", out / "model_int8.onnx"
    export_onnx(model, fp32_path, prep.crop)
    quantize_int8(fp32_path, int8_path, list(iter_batches(quant_paths, 1, prep)))
    fp32_path.unlink()  # only the served INT8 model is benchmarked; saves disk

    row = {
        "source": "this ablation (retrained memory bank, same seed and split)",
        "memory_bank_size": int(bank.shape[0]),
        "n_patches": int(patches.shape[0]),
        "coreset_seconds": round(coreset_s, 2),
        "bank_equals_prefix_of_default_bank": prefix_diff == 0.0,
        "max_abs_diff_to_default_bank_prefix": prefix_diff,
        "threshold": threshold,
        "int8": evaluate_int8(out, test, params, threshold),
        "int8_file_mb": round(int8_path.stat().st_size / 1e6, 1),
        **speed(out, test, params, f"{category}_{ratio}"),
    }
    return row


def main() -> None:
    """Run the ablation for the requested categories and ratios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "hazelnut", "screw", "carpet"]
    )
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.05, 0.01])
    args = parser.parse_args()

    params = load_params()
    tp, prep = params.train, params.preprocess
    torch.manual_seed(tp.seed)
    torch.set_num_threads(os.cpu_count() or 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "batch_size": 1,
        "runs": params.benchmark.runs,
        "warmup": params.benchmark.warmup,
    }
    tracking.setup(EXPERIMENT)

    for cat in args.categories:
        t_cat = time.perf_counter()
        meta0 = json.loads((PROJECT_ROOT / "models" / cat / "meta.json").read_text())
        calib_names = set(meta0["calib_images"])
        train = list_train(params.data_root, cat)
        bank_paths = [p for p in train if p.name not in calib_names]
        calib_paths = [p for p in train if p.name in calib_names]
        if (bank_paths, calib_paths) != split_train_calib(train, tp.calib_fraction, tp.seed):
            raise SystemExit(f"{cat}: calibration split differs from the main pipeline")
        quant_paths = list(bank_paths)
        random.Random(tp.seed).shuffle(quant_paths)  # same as scripts/export_onnx.py
        quant_paths = quant_paths[:N_QUANT_CALIB]
        test = TestSet(params, cat)
        default_bank = torch.load(
            PROJECT_ROOT / "models" / cat / "patchcore.pt", map_location="cpu", weights_only=True
        )["memory_bank"]

        rows = {DEFAULT_RATIO: default_row(params, cat, test)}
        embedder = build_embedder(params.model, prep)
        t0 = time.perf_counter()
        patches = extract_patches(embedder, iter_batches(bank_paths, tp.batch_size, prep))
        embed_s = time.perf_counter() - t0
        for ratio in args.ratios:
            rows[ratio] = ablation_row(
                params, cat, ratio, patches, embedder, calib_paths, quant_paths, test, default_bank
            )
            rows[ratio]["embedding_seconds"] = round(embed_s, 2)
        del patches

        for ratio, row in rows.items():
            result = {"category": cat, "coreset_ratio": ratio, **env, **row}
            (OUT_DIR / f"{cat}_{ratio}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            with mlflow.start_run(run_name=f"{cat}-r{ratio}"):
                mlflow.set_tags(
                    {"category": cat, "git_commit": tracking.git_commit(), "source": row["source"]}
                )
                mlflow.log_params({"coreset_ratio": ratio, "category": cat})
                flat = {
                    "memory_bank_size": row["memory_bank_size"],
                    "int8_file_mb": row["int8_file_mb"],
                    "int8_median_ms": row["int8_latency"]["median_ms"],
                    "int8_p95_ms": row["int8_latency"]["p95_ms"],
                    "nn_search_share": row["profile"]["nn_search_share"],
                    **{f"int8_{k}": v for k, v in row["int8"].items()},
                }
                mlflow.log_metrics(flat)
            print(
                f"{cat} r={ratio}: bank {row['memory_bank_size']}, "
                f"image AUROC {row['int8']['image_auroc']:.4f}, "
                f"AUPRO {row['int8']['aupro']:.4f}, "
                f"median {row['int8_latency']['median_ms']} ms, "
                f"NN share {row['profile']['nn_search_share']:.2f}",
                flush=True,
            )
        print(f"{cat} done in {time.perf_counter() - t_cat:.0f}s", flush=True)


if __name__ == "__main__":
    main()
