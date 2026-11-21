"""MVTec AD folder layout: listing train/test images and ground-truth masks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

CATEGORIES: tuple[str, ...] = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)  # fmt: skip

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class TestSample:
    """One test image with its label (0 = good, 1 = defect) and optional mask path."""

    path: Path
    label: int
    defect_type: str
    mask_path: Path | None


def _images(folder: Path) -> list[Path]:
    """Sorted image files in a folder (empty list if it does not exist)."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def list_train(root: Path, category: str) -> list[Path]:
    """All good training images of a category."""
    return _images(root / category / "train" / "good")


def list_test(root: Path, category: str) -> list[TestSample]:
    """All test images of a category, with labels and mask paths."""
    test_dir = root / category / "test"
    if not test_dir.is_dir():
        return []
    samples: list[TestSample] = []
    for defect_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        defect = defect_dir.name
        for img in _images(defect_dir):
            if defect == "good":
                samples.append(TestSample(img, 0, defect, None))
            else:
                mask = root / category / "ground_truth" / defect / f"{img.stem}_mask.png"
                samples.append(TestSample(img, 1, defect, mask if mask.exists() else None))
    return samples


def split_train_calib(
    paths: list[Path], calib_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Split good training images into a memory-bank set and a calibration set.

    The calibration images are never added to the memory bank; their scores set
    the decision threshold, so no defect labels are needed.
    """
    if not 0.0 <= calib_fraction < 1.0:
        raise ValueError("calib_fraction must be in [0, 1)")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    n_calib = round(len(shuffled) * calib_fraction)
    if calib_fraction > 0 and n_calib == 0 and len(shuffled) > 1:
        n_calib = 1
    calib = sorted(shuffled[:n_calib])
    train = sorted(shuffled[n_calib:])
    return train, calib
