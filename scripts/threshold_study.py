"""Threshold study (analysis only; the served threshold stays label-free).

1. Scores the calibration images listed in models/<category>/meta.json and all test
   images once with the served INT8 models -> results/threshold_study/scores.csv.
2. Evaluates the served threshold and four label-free rules on held-out good scores
   (p99, max, p95, mean + 3 std) for every category.
3. Compares calibration good scores with test good scores (median, p95, KS test).
4. Reference rows that USE TEST LABELS and are NOT used by the served model:
   few labels (best-F1 threshold on 5 good + 5 defective test images, evaluated on the
   rest, 20 random splits) and the oracle (best F1 on the full test set), plus the best
   recall reachable at <= 5% false positives.
5. Recall per defect type at the served threshold.

Run:  python scripts/threshold_study.py          (re-score and analyse)
      python scripts/threshold_study.py --reuse  (analyse the saved scores.csv only)
Writes results/threshold_study/scores.csv and summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import ks_2samp

from anomaly.config import PROJECT_ROOT, Params, load_params
from anomaly.data import list_test, list_train
from anomaly.metrics import image_auroc
from anomaly.onnx_infer import load_onnx_model
from anomaly.patchcore import iter_batches
from anomaly.thresholds import (
    LABEL_FREE_RULES,
    best_f1_threshold,
    decision_metrics,
    few_label_study,
    label_free_thresholds,
    recall_at_fpr,
)

OUT = PROJECT_ROOT / "results" / "threshold_study"
FIELDS = ("category", "split", "image", "defect_type", "label", "score")
MAX_FPR = 0.05


def score_all(params: Params) -> list[dict]:
    """Score calibration and test images of every category with the served INT8 model."""
    rows: list[dict] = []
    for cat in params.categories:
        model_dir = PROJECT_ROOT / "models" / cat
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        model = load_onnx_model(model_dir, "int8")
        calib_names = set(meta["calib_images"])
        calib = [p for p in list_train(params.data_root, cat) if p.name in calib_names]
        test = list_test(params.data_root, cat)
        items = [("calib", p, "good", 0) for p in calib]
        items += [("test", s.path, s.defect_type, s.label) for s in test]
        paths = [it[1] for it in items]
        scores = np.concatenate(
            [
                model.predict(b)[0]
                for b in iter_batches(paths, params.evaluate.batch_size, params.preprocess)
            ]
        )
        for (split, path, defect, label), score in zip(items, scores, strict=True):
            rows.append({
                "category": cat,
                "split": split,
                "image": path.relative_to(params.data_root / cat).as_posix(),
                "defect_type": defect,
                "label": label,
                "score": float(score),
            })  # fmt: skip
        print(f"scored {cat}: {len(calib)} calibration, {len(test)} test images", flush=True)
    return rows


def analyse_category(rows: list[dict], served_threshold: float) -> dict:
    """All threshold numbers for one category."""
    calib = np.array([r["score"] for r in rows if r["split"] == "calib"])
    test = [r for r in rows if r["split"] == "test"]
    labels = np.array([r["label"] for r in test])
    scores = np.array([r["score"] for r in test])
    good = scores[labels == 0]

    def metrics(t: float) -> dict:
        m = decision_metrics(labels, scores, t)
        return {"threshold": t, **{k: m[k] for k in ("precision", "recall", "fpr", "f1")}}

    rules = {"served": metrics(served_threshold)}
    rules.update({k: metrics(v) for k, v in label_free_thresholds(calib).items()})
    ks = ks_2samp(calib, good)
    oracle_t = best_f1_threshold(labels, scores)
    per_defect = defaultdict(list)
    for r in test:
        if r["label"] == 1:
            per_defect[r["defect_type"]].append(r["score"] > served_threshold)
    return {
        "n_calib": len(calib),
        "n_test_good": int((labels == 0).sum()),
        "n_test_defect": int(labels.sum()),
        "image_auroc_int8": image_auroc(labels, scores),
        "rules": rules,
        "shift": {
            "calib_good_median": float(np.median(calib)),
            "calib_good_p95": float(np.percentile(calib, 95)),
            "test_good_median": float(np.median(good)),
            "test_good_p95": float(np.percentile(good, 95)),
            "median_ratio_test_to_calib": float(np.median(good) / np.median(calib)),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        },
        "uses_test_labels": {
            "few_label_5_plus_5": few_label_study(labels, scores, 5, 5, 20, seed=0),
            "oracle_best_f1": metrics(oracle_t),
            f"recall_at_fpr_{MAX_FPR}": recall_at_fpr(labels, scores, MAX_FPR),
        },
        "recall_per_defect_type_served": {
            d: {"recall": float(np.mean(v)), "n": len(v)} for d, v in sorted(per_defect.items())
        },
    }


def main() -> None:
    """Score (unless --reuse) and write the study summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse", action="store_true", help="analyse the saved scores.csv")
    args = parser.parse_args()
    params = load_params()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "scores.csv"

    if args.reuse:
        with csv_path.open(encoding="utf-8") as f:
            rows = [
                {**r, "label": int(r["label"]), "score": float(r["score"])}
                for r in csv.DictReader(f)
            ]
    else:
        rows = score_all(params)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    categories = {}
    for cat, cat_rows in by_cat.items():
        meta = json.loads((PROJECT_ROOT / "models" / cat / "meta.json").read_text())
        categories[cat] = analyse_category(cat_rows, float(meta["threshold"]))

    def mean_rule(rule: str, key: str) -> float:
        return float(np.mean([c["rules"][rule][key] for c in categories.values()]))

    summary = {
        "model": "served INT8 ONNX models (models/<category>/model_int8.onnx)",
        "note": "Rows under 'uses_test_labels' use test labels and are NOT used by the "
        "served model; they only show how much a threshold could improve.",
        "mean_over_categories": {
            rule: {k: mean_rule(rule, k) for k in ("precision", "recall", "fpr", "f1")}
            for rule in ("served", *LABEL_FREE_RULES)
        },
        "categories": categories,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path} and {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
