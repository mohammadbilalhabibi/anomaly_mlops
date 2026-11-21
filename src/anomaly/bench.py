"""Latency measurement helpers (median / p95 per call)."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Sequence

import numpy as np


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


def summarize(ms: Sequence[float]) -> dict[str, float]:
    """Median, p95 and mean of millisecond timings."""
    arr = np.asarray(ms, dtype=np.float64)
    return {
        "median_ms": round(float(np.median(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
    }


def time_calls(
    infer: Callable[[np.ndarray], object], inputs: Sequence[np.ndarray], warmup: int
) -> list[float]:
    """Time `infer` on each input after `warmup` untimed calls; returns ms per call."""
    for x in inputs[:warmup]:
        infer(x)
    times = []
    for x in inputs:
        t0 = time.perf_counter()
        infer(x)
        times.append((time.perf_counter() - t0) * 1000)
    return times
