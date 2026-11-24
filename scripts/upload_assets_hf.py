"""Upload the serving models and dashboard samples to a PRIVATE Hugging Face model repo.

The deployed container downloads them at startup (see src/anomaly/serve/assets.py), so the
public Docker image never contains MVTec-derived files (CC BY-NC-SA 4.0).

Uploaded per category: models/<cat>/model_<variant>.onnx and meta.json, plus the same sample
images the dashboard shows locally (index_samples, 6 per category by default).

Needs a Hugging Face login with a WRITE token (`hf auth login`); run once and after retraining:
    python scripts/upload_assets_hf.py --repo <user>/anomaly-mlops-assets
    python scripts/upload_assets_hf.py --repo <user>/anomaly-mlops-assets --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from anomaly.config import PROJECT_ROOT
from anomaly.onnx_infer import VARIANT_FILES
from anomaly.serve.api import index_samples

MODEL_CARD = """---
license: cc-by-nc-sa-4.0
tags: [anomaly-detection, patchcore, onnx, mvtec-ad]
---
# anomaly_mlops serving assets (private)

PatchCore ONNX models ({variant}) and dashboard sample images for the anomaly_mlops demo
(https://github.com/mohammadbilalhabibi/anomaly_mlops).

The models are trained on, and the sample images are taken from, the MVTec AD dataset
(MVTec Software GmbH, https://www.mvtec.com/company/research/datasets/mvtec-ad), licensed
CC BY-NC-SA 4.0. Non-commercial use only; this repository keeps the same licence.
"""


def stage_assets(
    models_dir: Path, data_dir: Path, variant: str, per_category: int, out: Path
) -> dict[str, int]:
    """Copy models, metadata and sample images into `out` in the repo layout; returns counts."""
    model_file = VARIANT_FILES[variant]
    categories = sorted(
        d.name
        for d in models_dir.iterdir()
        if (d / model_file).is_file() and (d / "meta.json").is_file()
    )
    for cat in categories:
        dest = out / "models" / cat
        dest.mkdir(parents=True, exist_ok=True)
        for name in (model_file, "meta.json"):
            shutil.copy2(models_dir / cat / name, dest / name)
    samples = index_samples(data_dir, categories, per_category)
    for s in samples.values():
        dest = out / "samples" / s.path.relative_to(data_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s.path, dest)
    (out / "README.md").write_text(MODEL_CARD.format(variant=variant), encoding="utf-8")
    return {"categories": len(categories), "samples": len(samples)}


def main() -> None:
    """Stage the assets and upload them to a private model repo (created if missing)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="<user>/<name> of the private model repo")
    parser.add_argument("--variant", default="int8", choices=sorted(VARIANT_FILES))
    parser.add_argument("--per-category", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="only stage and list the files")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        counts = stage_assets(
            PROJECT_ROOT / "models",
            PROJECT_ROOT / "data" / "mvtec_ad",
            args.variant,
            args.per_category,
            out,
        )
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        n_models, n_samples = counts["categories"], counts["samples"]
        print(f"staged {n_models} models, {n_samples} samples, {size / 1e9:.2f} GB")
        if args.dry_run:
            return
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
        if not api.repo_info(args.repo, repo_type="model").private:
            raise SystemExit(f"{args.repo} exists and is PUBLIC; refusing to upload MVTec assets")
        api.upload_folder(
            repo_id=args.repo,
            repo_type="model",
            folder_path=out,
            commit_message=f"Upload {args.variant} models and dashboard samples",
        )
        print(f"uploaded to https://huggingface.co/{args.repo} (private)")


if __name__ == "__main__":
    main()
