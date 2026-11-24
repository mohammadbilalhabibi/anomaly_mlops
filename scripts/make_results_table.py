"""Generate the results tables from results/*.json (never typed by hand).

Writes results/RESULTS.md and replaces each generated block in README.md: the text
between <!-- NAME:START --> and <!-- NAME:END --> for NAME in HEADLINE, CORESET,
THRESHOLD and RESULTS.
"""

from __future__ import annotations

import json
from pathlib import Path

from anomaly.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"


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


def headline_table() -> str:
    """Small first-screen table: PyTorch vs served INT8, speed and size, API latency."""
    torch_r, int8_r = _load(RESULTS / "torch"), _load(RESULTS / "int8")
    lat, exp = _load(RESULTS / "latency"), _load(RESULTS / "export")
    if not torch_r or not int8_r:
        return "_No results yet. Run `dvc repro`._"
    n = len(torch_r)

    def mean(rs: dict, k: str) -> str:
        return f"{_pct(sum(r[k] for r in rs.values()) / len(rs))}%"

    lines = [
        f"| Mean over {n} MVTec AD categories | PyTorch (reference) | ONNX INT8 (served) |",
        "|---|---:|---:|",
        f"| Image AUROC (finds the defective part) | {mean(torch_r, 'image_auroc')} "
        f"| {mean(int8_r, 'image_auroc')} |",
        f"| Pixel AUROC (finds the defect pixels) | {mean(torch_r, 'pixel_auroc')} "
        f"| {mean(int8_r, 'pixel_auroc')} |",
        f"| AUPRO (FPR ≤ 0.3, per defect region) | {mean(torch_r, 'aupro')} "
        f"| {mean(int8_r, 'aupro')} |",
    ]
    if lat:
        speedup = sum(
            r["onnx_fp32"]["median_ms"] / r["onnx_int8"]["median_ms"] for r in lat.values()
        ) / len(lat)
        meds = sorted(r["onnx_int8"]["median_ms"] for r in lat.values())
        lines.append(
            f"| Median CPU latency per image, model only | – | {meds[0]:.0f}–{meds[-1]:.0f} ms |"
        )
        lines.append(f"| Speed vs ONNX fp32 (median latency) | – | {speedup:.2f}× faster |")
    if exp:
        size = sum(r["int8"]["file_mb"] / r["fp32"]["file_mb"] for r in exp.values()) / len(exp)
        lines.append(f"| Model file size vs ONNX fp32 | – | {100 * (1 - size):.0f}% smaller |")
    api = RESULTS / "latency_api.json"
    if api.is_file():
        a = json.loads(api.read_text(encoding="utf-8"))
        lines.append(
            f"| API end-to-end, {a['category']} (median / p95) | – "
            f"| {a['median_ms']:.0f} / {a['p95_ms']:.0f} ms |"
        )
    passed = sum(1 for r in int8_r.values() if r.get("quality_gate", {}).get("passed"))
    lines += [
        "",
        f"INT8 passes the quality gate (at most 2 points lost on every metric) in "
        f"{passed}/{len(int8_r)} categories. Latency: laptop CPU, batch size 1.",
    ]
    return "\n".join(lines)


def _coreset_rows() -> list[dict]:
    rows = list(_load(RESULTS / "ablation_coreset").values())
    order = {c: i for i, c in enumerate(dict.fromkeys(r["category"] for r in rows))}
    return sorted(rows, key=lambda r: (order[r["category"]], -r["coreset_ratio"]))


def coreset_block() -> str:
    """Coreset ablation: memory-bank size vs accuracy, latency and NN-search share."""
    rows = _coreset_rows()
    if not rows:
        return "_Not run yet. Run `python scripts/ablate_coreset.py`._"
    first = rows[0]
    by_ratio: dict[float, list[dict]] = {}
    for r in rows:
        by_ratio.setdefault(r["coreset_ratio"], []).append(r)
    base = {r["category"]: r["int8_latency"]["median_ms"] for r in by_ratio.get(0.1, [])}
    lines = [
        f"Mean over {len(by_ratio.get(0.1, rows))} categories ({', '.join(base) or '–'}), "
        "served INT8 model:",
        "",
        "| Coreset ratio | Memory bank (mean) | Image AUROC | Pixel AUROC | AUPRO "
        "| INT8 median ms | Speed-up vs 10% | NN-search share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ratio in sorted(by_ratio, reverse=True):
        rs = by_ratio[ratio]

        def m(key: str, rs: list[dict] = rs) -> float:
            return sum(r["int8"][key] for r in rs) / len(rs)

        bank = sum(r["memory_bank_size"] for r in rs) / len(rs)
        med = sum(r["int8_latency"]["median_ms"] for r in rs) / len(rs)
        sp = [
            base[r["category"]] / r["int8_latency"]["median_ms"]
            for r in rs
            if r["category"] in base
        ]
        share = sum(r["profile"]["nn_search_share"] for r in rs) / len(rs)
        speed = f"{sum(sp) / len(sp):.2f}×" if sp else "–"
        lines.append(
            f"| {100 * ratio:g}% | {bank:,.0f} | {_pct(m('image_auroc'))} "
            f"| {_pct(m('pixel_auroc'))} | {_pct(m('aupro'))} | {med:.0f} | {speed} "
            f"| {100 * share:.0f}% |"
        )
    lines += [
        "",
        "<details><summary>Per category and ratio</summary>",
        "",
        "| Category | Coreset | Memory bank | Image AUROC | Pixel AUROC | AUPRO | Precision "
        "| Recall | FPR | INT8 MB | INT8 median / p95 ms | NN-search share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        q, lat = r["int8"], r["int8_latency"]
        lines.append(
            f"| {r['category']} | {100 * r['coreset_ratio']:g}% | {r['memory_bank_size']:,} "
            f"| {_pct(q['image_auroc'])} | {_pct(q['pixel_auroc'])} | {_pct(q['aupro'])} "
            f"| {_pct(q['precision_at_threshold'])} | {_pct(q['recall_at_threshold'])} "
            f"| {_pct(q['false_positive_rate_at_threshold'])} | {r['int8_file_mb']} "
            f"| {lat['median_ms']:.0f} / {lat['p95_ms']:.0f} "
            f"| {100 * r['profile']['nn_search_share']:.0f}% |"
        )
    lines += ["", "</details>", ""]
    lines.append(
        f"Latency: {first['cpu']}, batch size 1, {first['runs']} runs after {first['warmup']} "
        "warm-up runs; all ratios were timed in the same session (the 10% rows reuse the main "
        "models and re-time them). Precision, recall and FPR use a threshold recalibrated for "
        "each ratio. NN-search share = share of ONNX Runtime kernel time spent comparing "
        "patches with the memory bank."
    )
    nested = [r for r in rows if "bank_equals_prefix_of_default_bank" in r]
    if nested:
        same = sum(1 for r in nested if r["bank_equals_prefix_of_default_bank"])
        lines.append(
            f"Reproducibility check: {same}/{len(nested)} retrained memory banks are exactly "
            "the first rows of the 10% bank (greedy coreset selection is nested)."
        )
    return "\n".join(lines)


def threshold_block() -> str:
    """Threshold study: label-free rules, score shift and label-based reference rows."""
    path = RESULTS / "threshold_study" / "summary.json"
    if not path.is_file():
        return "_Not run yet. Run `python scripts/threshold_study.py`._"
    st = json.loads(path.read_text(encoding="utf-8"))
    cats = st["categories"]
    names = {
        "served": "Served (p99, calibrated in PyTorch)",
        "p99": "p99",
        "max": "max",
        "p95": "p95",
        "mean_3std": "mean + 3·std",
    }
    lines = [
        f"Mean over {len(cats)} categories, served INT8 models (%):",
        "",
        "| Threshold rule | Uses defect labels? | Precision | Recall | FPR | F1 |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    for rule, m in st["mean_over_categories"].items():
        lines.append(
            f"| {names.get(rule, rule)} | no | {_pct(m['precision'])} | {_pct(m['recall'])} "
            f"| {_pct(m['fpr'])} | {_pct(m['f1'])} |"
        )
    few = [c["uses_test_labels"]["few_label_5_plus_5"] for c in cats.values()]
    orc = [c["uses_test_labels"]["oracle_best_f1"] for c in cats.values()]

    def avg(rows: list[dict], k: str, sub: str | None = None) -> float:
        return sum((r[k][sub] if sub else r[k]) for r in rows) / len(rows)

    lines.append(
        "| *Reference:* best F1 on 5 good + 5 defective test images (20 splits) | **yes** "
        f"| {_pct(avg(few, 'precision', 'mean'))} | {_pct(avg(few, 'recall', 'mean'))} "
        f"| {_pct(avg(few, 'fpr', 'mean'))} | {_pct(avg(few, 'f1', 'mean'))} |"
    )
    lines.append(
        "| *Reference:* oracle, best F1 on the full test set | **yes** "
        f"| {_pct(avg(orc, 'precision'))} | {_pct(avg(orc, 'recall'))} "
        f"| {_pct(avg(orc, 'fpr'))} | {_pct(avg(orc, 'f1'))} |"
    )
    lines += [
        "",
        "Rows marked *Reference* use test labels and are not used by the served model.",
        "",
        "<details><summary>Per category: served threshold, best possible threshold and "
        "good-score shift</summary>",
        "",
        "| Category | Calib. images | Served recall | Served FPR | Best recall at FPR ≤ 5% "
        "(labels) | Oracle F1 (labels) | Few-label F1 (labels) | Good-score median, "
        "test / calib. | KS p-value |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cat, c in cats.items():
        sv, ul, sh = c["rules"]["served"], c["uses_test_labels"], c["shift"]
        fl = ul["few_label_5_plus_5"]["f1"]
        rk = next(k for k in ul if k.startswith("recall_at_fpr"))
        lines.append(
            f"| {cat} | {c['n_calib']} | {_pct(sv['recall'])} | {_pct(sv['fpr'])} "
            f"| {_pct(ul[rk])} | {_pct(ul['oracle_best_f1']['f1'])} "
            f"| {_pct(fl['mean'])} ± {_pct(fl['std'])} "
            f"| {sh['median_ratio_test_to_calib']:.2f}× | {sh['ks_pvalue']:.2g} |"
        )
    lines += ["", "</details>"]
    hard = [c for c in ("screw", "pill", "capsule") if c in cats]
    if hard:
        lines += [
            "",
            "Recall per defect type at the served threshold (%), lowest first:",
            "",
            "| Category | Defect types: recall (n) |",
            "|---|---|",
        ]
        for cat in hard:
            per = sorted(
                cats[cat]["recall_per_defect_type_served"].items(),
                key=lambda kv: kv[1]["recall"],
            )
            cells = ", ".join(f"{d} {_pct(v['recall'])} ({v['n']})" for d, v in per)
            lines.append(f"| {cat} | {cells} |")
    return "\n".join(lines)


def build() -> str:
    """Assemble the per-category tables into one markdown block."""
    parts = [
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


def deploy_block() -> str:
    """Measured latency and cold start of the deployed app (results/latency_azure.json)."""
    path = RESULTS / "latency_azure.json"
    if not path.is_file():
        return "_Not measured yet: run `scripts/benchmark_api.py --out latency_azure.json`._"
    a = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        f"| Live app ({a['category']}, ONNX {a['variant']}, {a['runs']} requests) | median | p95 |",
        "|---|---:|---:|",
        f"| Model only (on the server) | {a['inference_median_ms']:.0f} ms | – |",
        f"| Server total: decode, model, heatmap PNGs | {a['server_median_ms']:.0f} ms "
        f"| {a['server_p95_ms']:.0f} ms |",
        f"| Round trip from a laptop in Deggendorf (incl. upload) | {a['median_ms']:.0f} ms "
        f"| {a['p95_ms']:.0f} ms |",
    ]
    if "cold_start_s" in a:
        lines += [
            "",
            f"First request after the app has scaled to zero: **{a['cold_start_s']:.0f} s** "
            "(start a replica, pull the image, download 1.5 GB of models and samples, load 15 "
            "models). Every request after that is fast.",
        ]
    return "\n".join(lines)


def blocks() -> dict[str, str]:
    """All generated README blocks, keyed by marker name."""
    return {
        "HEADLINE": headline_table() + "\n",
        "CORESET": coreset_block() + "\n",
        "THRESHOLD": threshold_block() + "\n",
        "DEPLOY": deploy_block() + "\n",
        "RESULTS": build(),
    }


def replace_blocks(text: str, generated: dict[str, str]) -> str:
    """Replace the content between <!-- NAME:START --> and <!-- NAME:END --> markers."""
    for name, block in generated.items():
        start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
        if start in text and end in text:
            head, rest = text.split(start, 1)
            _, tail = rest.split(end, 1)
            text = f"{head}{start}\n{block}{end}{tail}"
    return text


def main() -> None:
    """Write results/RESULTS.md and update README.md in place."""
    gen = blocks()
    titles = {
        "HEADLINE": "Headline",
        "CORESET": "Coreset ablation (speed vs accuracy)",
        "THRESHOLD": "Threshold study",
        "DEPLOY": "Live deployment (Azure Container Apps)",
        "RESULTS": "Per-category results",
    }
    md = "# Measured results\n\n" + "\n".join(f"## {titles[k]}\n\n{v}" for k, v in gen.items())
    (RESULTS / "RESULTS.md").write_text(md, encoding="utf-8")
    readme = PROJECT_ROOT / "README.md"
    readme.write_text(replace_blocks(readme.read_text(encoding="utf-8"), gen), encoding="utf-8")
    print(f"Wrote {RESULTS / 'RESULTS.md'} and updated README.md")


if __name__ == "__main__":
    main()
