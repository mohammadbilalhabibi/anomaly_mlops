"""Check that the MVTec AD categories in params.yaml are present and complete.

Writes results/data_report.json with image counts per category (DVC metric).
"""

from __future__ import annotations

import argparse
import json
import sys

from anomaly.config import PROJECT_ROOT, load_params
from anomaly.data import list_test, list_train


def main() -> int:
    """Count train/test images and masks; return a non-zero exit code on problems."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "results" / "data_report.json"))
    args = parser.parse_args()

    params = load_params()
    report: dict[str, dict[str, int]] = {}
    problems: list[str] = []
    for cat in params.categories:
        train = list_train(params.data_root, cat)
        test = list_test(params.data_root, cat)
        n_defect = sum(s.label for s in test)
        n_masks = sum(1 for s in test if s.mask_path is not None)
        report[cat] = {
            "train_good": len(train),
            "test_good": len(test) - n_defect,
            "test_defect": n_defect,
            "masks": n_masks,
            "defect_types": len({s.defect_type for s in test if s.label == 1}),
        }
        if not train or not test:
            problems.append(f"{cat}: missing images under {params.data_root / cat}")
        elif n_masks != n_defect:
            problems.append(f"{cat}: {n_defect - n_masks} defect images without a mask")
        print(f"{cat:12s} {report[cat]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
