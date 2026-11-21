"""Generate the results tables from results/*.json (never typed by hand).

Writes results/RESULTS.md and replaces the block between
<!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from anomaly.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _load(folder: Path) -> dict[str, dict]:
    """Load every <category>.json in a folder, keyed by category."""
    if not folder.is_dir():
        return {}
    return {
        p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))
    }


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}"


def _gate(r: dict) -> str:
    gate = r.get("quality_gate")
    if gate is None:
        return "–"
    return "pass" if gate["passed"] else "**fail**"


def accuracy_table() -> str:
    """Per-category image AUROC, pixel AUROC and AUPRO: PyTorch reference and served INT8."""
    torch_r, fp32_r, int8_r = (_load(RESULTS / b) for b in ("torch", "fp32", "int8"))
    if not torch_r:
        return "_No results yet. Run `dvc repro`._"
    keys = ("image_auroc", "pixel_auroc", "aupro")
    lines = [
        "| Category | Image AUROC | Pixel AUROC | AUPRO (FPR≤0.3) "
        "| INT8 Image AUROC | INT8 Pixel AUROC | INT8 AUPRO | INT8 gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    sums = {b: dict.fromkeys(keys, 0.0) for b in ("torch", "int8")}
    for cat, r in torch_r.items():
        q = int8_r.get(cat, {})
        lines.append(
            f"| {cat} | "
            + " | ".join(_pct(r[k]) for k in keys)
            + " | "
            + " | ".join(_pct(q.get(k)) for k in keys)
            + f" | {_gate(q)} |"
        )
        for k in keys:
            sums["torch"][k] += r[k]
            sums["int8"][k] += q.get(k, float("nan"))
    n = len(torch_r)
    if n > 1:
        lines.append(
            f"| **mean ({n})** | "
            + " | ".join(f"**{_pct(sums['torch'][k] / n)}**" for k in keys)
            + " | "
            + " | ".join(f"**{_pct(sums['int8'][k] / n)}**" for k in keys)
            + " | |"
        )
    shared = [c for c in torch_r if c in fp32_r]
    if shared:
        diff = max(abs(torch_r[c][k] - fp32_r[c][k]) for c in shared for k in keys)
        lines += [
            "",
            f"PyTorch columns = reference model. ONNX fp32 differs from it by at most "
            f"{100 * diff:.2g} points on any metric across {len(shared)} categories. "
            "INT8 gate: every metric within the `max_metric_drop` set in `params.yaml`.",
        ]
    return "\n".join(lines)


def threshold_table() -> str:
    """Decisions at the label-free threshold (99th percentile of held-out good scores)."""
    torch_r = _load(RESULTS / "torch")
    if not torch_r:
        return ""
    lines = [
        "| Category | Test images (defect) | Precision | Recall | False-positive rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, r in torch_r.items():
        lines.append(
            f"| {cat} | {r['n_test']} ({r['n_defect']}) | {_pct(r['precision_at_threshold'])} "
            f"| {_pct(r['recall_at_threshold'])} | {_pct(r['false_positive_rate_at_threshold'])} |"
        )
    return "\n".join(lines)


def latency_table() -> str:
    """Median / p95 latency per image for each backend."""
    lat = _load(RESULTS / "latency")
    if not lat:
        return "_No latency measured yet. Run `dvc repro`._"
    first = next(iter(lat.values()))
    lines = [
        f"Measured on {first['cpu']} ({first['logical_cores']} logical cores, no GPU), "
        f"batch size 1, {first['runs']} runs after {first['warmup']} warm-up runs, "
        f"torch {first['torch_version']}, onnxruntime {first['onnxruntime_version']}. "
        "Times in ms per image (median / p95).",
        "",
        "| Category | Memory bank | Preprocess | PyTorch fp32 | ONNX fp32 | ONNX INT8 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat, r in lat.items():
        cells = [
            f"{r[k]['median_ms']:.1f} / {r[k]['p95_ms']:.1f}"
            for k in ("preprocess", "torch", "onnx_fp32", "onnx_int8")
        ]
        lines.append(f"| {cat} | {r['memory_bank_size']:,} | " + " | ".join(cells) + " |")
    api = RESULTS / "latency_api.json"
    if api.is_file():
        a = json.loads(api.read_text(encoding="utf-8"))
        lines += [
            "",
            f"End-to-end HTTP `POST /api/predict` ({a['category']}, ONNX {a['variant']}, "
            f"{a['runs']} requests on localhost, incl. upload, decode, heatmap PNG encoding): "
            f"median {a['median_ms']:.1f} ms, p95 {a['p95_ms']:.1f} ms.",
        ]
    return "\n".join(lines)


def export_table() -> str:
    """ONNX file sizes and score parity against PyTorch."""
    exp = _load(RESULTS / "export")
    if not exp:
        return ""
    lines = [
        "| Category | ONNX fp32 MB | ONNX INT8 MB | fp32 max rel. score diff "
        "| INT8 max rel. score diff |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, r in exp.items():
        lines.append(
            f"| {cat} | {r['fp32']['file_mb']} | {r['int8']['file_mb']} "
            f"| {r['fp32']['max_rel_score_diff']:.1e} | {r['int8']['max_rel_score_diff']:.1e} |"
        )
    return "\n".join(lines)


def quantization_table() -> str:
    """Localisation with the blur convolutions quantized vs. kept in fp32 (measured ablation)."""
    blur_q = _load(RESULTS / "ablation_int8_blur_quantized")
    fp32_r, int8_r = _load(RESULTS / "fp32"), _load(RESULTS / "int8")
    if not blur_q:
        return ""
    lines = [
        "Pixel AUROC / AUPRO (%) on the test set for three ONNX variants:",
        "",
        "| Category | fp32 | INT8, blur convs also quantized | INT8, backbone only (served) |",
        "|---|---:|---:|---:|",
    ]
    for cat, q in blur_q.items():
        f, i = fp32_r.get(cat, {}), int8_r.get(cat, {})

        def cell(r: dict) -> str:
            return f"{_pct(r.get('pixel_auroc'))} / {_pct(r.get('aupro'))}"

        lines.append(f"| {cat} | {cell(f)} | {cell(q)} | {cell(i)} |")
    return "\n".join(lines)


def summary() -> str:
    """One-paragraph headline computed from the result files."""
    torch_r, int8_r = _load(RESULTS / "torch"), _load(RESULTS / "int8")
    lat = _load(RESULTS / "latency")
    if not torch_r:
        return ""
    n = len(torch_r)

    def mean(rs: dict, k: str) -> str:
        return _pct(sum(r[k] for r in rs.values()) / len(rs))

    text = (
        f"**Headline ({n} categories):** mean image AUROC {mean(torch_r, 'image_auroc')}%, "
        f"pixel AUROC {mean(torch_r, 'pixel_auroc')}%, AUPRO {mean(torch_r, 'aupro')}%."
    )
    if int8_r:
        passed = sum(1 for r in int8_r.values() if r.get("quality_gate", {}).get("passed"))
        text += (
            f" The served INT8 ONNX model keeps image AUROC {mean(int8_r, 'image_auroc')}% / "
            f"AUPRO {mean(int8_r, 'aupro')}% and passes the quality gate in {passed}/{len(int8_r)} "
            "categories."
        )
    if lat:
        meds = sorted(r["onnx_int8"]["median_ms"] for r in lat.values())
        text += (
            f" INT8 inference takes {meds[0]:.0f}–{meds[-1]:.0f} ms median per image on a "
            "laptop CPU, depending on memory-bank size."
        )
    return text


def build() -> str:
    """Assemble all tables into one markdown block."""
    parts = [
        summary(),
        "### Detection and localisation (MVTec AD test set, %)",
        accuracy_table(),
        "### Decisions at the label-free threshold (%)",
        threshold_table(),
        "### CPU latency per image",
        latency_table(),
        "### ONNX export",
        export_table(),
        "### What INT8 quantization should not touch",
        quantization_table(),
    ]
    return "\n\n".join(p for p in parts if p) + "\n"


def main() -> None:
    """Write results/RESULTS.md and update README.md in place."""
    block = build()
    (RESULTS / "RESULTS.md").write_text("# Measured results\n\n" + block, encoding="utf-8")
    readme = PROJECT_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START in text and END in text:
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        readme.write_text(f"{head}{START}\n{block}{END}{tail}", encoding="utf-8")
    print(f"Wrote {RESULTS / 'RESULTS.md'} and updated README.md")


if __name__ == "__main__":
    main()
