"""Measure end-to-end HTTP latency of a running API (POST /api/predict).

Start the server first (`uvicorn anomaly.serve.api:app`), then run:
    python scripts/benchmark_api.py --category bottle
Writes results/latency_api.json.
"""

from __future__ import annotations

import argparse
import json
import time

import httpx
import numpy as np

from anomaly.config import PROJECT_ROOT, load_params
from anomaly.data import list_test


def main() -> None:
    """Send test images to the API and record median / p95 request latency."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    params = load_params()
    paths = [s.path for s in list_test(params.data_root, args.category)]
    payloads = [paths[i % len(paths)].read_bytes() for i in range(args.runs + args.warmup)]
    times = []
    with httpx.Client(base_url=args.url, timeout=60) as client:
        variant = client.get("/api/health").json()["variant"]
        for i, data in enumerate(payloads):
            t0 = time.perf_counter()
            r = client.post(
                "/api/predict",
                data={"category": args.category},
                files={"file": ("img.png", data, "image/png")},
            )
            r.raise_for_status()
            if i >= args.warmup:
                times.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(times)
    report = {
        "category": args.category,
        "variant": variant,
        "runs": args.runs,
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }
    (PROJECT_ROOT / "results" / "latency_api.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
