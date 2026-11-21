"""Where does inference time go? ONNX Runtime profiling split by graph stage.

Stages of the exported PatchCore graph:
- backbone: everything up to and including the channel projection (features per patch);
- nn_search: distance of every patch to every memory-bank entry and the minimum;
- postprocess: image score, upsampling and Gaussian blur of the anomaly map.

Shares are computed from per-operator kernel times reported by ONNX Runtime
(not wall-clock time), after discarding warm-up runs.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

STAGES = ("backbone", "nn_search", "postprocess")
_TOKEN_SUFFIX = re.compile(r"_token_\d+$")


def node_stages(model_path: Path) -> dict[str, str]:
    """Map every node name of a PatchCore ONNX graph to its stage."""
    graph = onnx.load(str(model_path), load_external_data=False).graph
    nodes = list(graph.node)
    producer = {out: n for n in nodes for out in n.output}
    consumers: dict[str, list] = defaultdict(list)
    for n in nodes:
        for i in n.input:
            consumers[i].append(n)

    def walk(start, step) -> set[str]:
        seen, stack = set(), [start]
        while stack:
            n = stack.pop()
            if n.name not in seen:
                seen.add(n.name)
                stack += step(n)
        return seen

    def parents(n) -> list:
        return [producer[i] for i in n.input if i in producer]

    def children(n) -> list:
        return [c for o in n.output for c in consumers[o]]

    proj = next(n for n in nodes if any(i.endswith("embedder.projection") for i in n.input))
    bank_mm = next(n for n in nodes if "scorer.bank_t" in n.input)
    after_bank = walk(bank_mm, children)
    sqrt = next(n for n in nodes if n.op_type == "Sqrt" and n.name in after_bank)
    backbone = walk(proj, parents)
    nn_search = (walk(proj, children) & walk(sqrt, parents)) - {proj.name}
    return {
        n.name: "backbone"
        if n.name in backbone
        else "nn_search"
        if n.name in nn_search
        else "postprocess"
        for n in nodes
    }


def stage_of_profiled(name: str, stages: dict[str, str]) -> str:
    """Stage of a profiled (possibly fused or renamed) node; unknown nodes count as backbone.

    ONNX Runtime keeps the original node name as a prefix when it fuses nodes
    ("node_matmul_1/MatMulScaleFusion/") and appends "_token_N" to quantized
    convolutions; names it invents for QDQ/quantized kernels belong to the backbone.
    """
    base = _TOKEN_SUFFIX.sub("", name.split("/")[0])
    return stages.get(base, stages.get(name, "backbone"))


def parse_profile(events: list[dict], stages: dict[str, str], warmup: int) -> dict[str, float]:
    """Sum kernel time per stage over the runs after `warmup`; returns ms per run and shares."""
    runs = sorted(
        e["ts"] for e in events if e.get("cat") == "Session" and e.get("name") == "model_run"
    )
    if len(runs) <= warmup:
        raise ValueError("not enough profiled runs after warm-up")
    start, n_runs = runs[warmup], len(runs) - warmup
    totals = dict.fromkeys(STAGES, 0.0)
    for e in events:
        if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time") or e["ts"] < start:
            continue
        totals[stage_of_profiled(e["name"][: -len("_kernel_time")], stages)] += e["dur"]
    total = sum(totals.values())
    out: dict[str, float] = {f"{s}_ms": round(totals[s] / 1000 / n_runs, 2) for s in STAGES}
    out.update({f"{s}_share": round(totals[s] / total, 4) for s in STAGES})
    out["profiled_runs"] = n_runs
    return out


def profile_stages(
    model_path: Path, inputs: Sequence[np.ndarray], warmup: int, prefix: Path
) -> dict[str, float]:
    """Run `inputs` through a profiling session and split kernel time by stage."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_profiling = True
    opts.profile_file_prefix = str(prefix)
    sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
    for x in inputs:
        sess.run(None, {"image": x})
    path = Path(sess.end_profiling())
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    finally:
        path.unlink(missing_ok=True)
    return parse_profile(events, node_stages(model_path), warmup)
