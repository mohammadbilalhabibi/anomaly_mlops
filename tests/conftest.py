"""Shared fixtures: a tiny synthetic MVTec-style dataset and a tiny PatchCore model.

Nothing here needs the real dataset or downloaded weights: images are generated,
and the backbone is a randomly initialised ResNet-18.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomaly.config import ModelParams, PreprocessParams

IMG = 80
PREP = PreprocessParams(resize=72, crop=64)
MODEL = ModelParams(
    backbone="resnet18", layers=("layer2", "layer3"), embed_dim=64, pool_size=3, blur_sigma=1.0
)


def good_image(rng: np.random.Generator) -> np.ndarray:
    """A smooth gradient texture with mild noise (a 'defect-free product')."""
    y, x = np.mgrid[0:IMG, 0:IMG] / IMG
    base = 0.5 + 0.2 * np.sin(6 * x) * np.cos(6 * y)
    img = np.stack([base, base * 0.9, base * 0.8], axis=-1) + rng.normal(0, 0.02, (IMG, IMG, 3))
    return (img.clip(0, 1) * 255).astype(np.uint8)


def defect_image(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A good image with a high-contrast square scratch, plus its mask."""
    img = good_image(rng)
    mask = np.zeros((IMG, IMG), dtype=np.uint8)
    r, c = rng.integers(20, 44, size=2)
    img[r : r + 16, c : c + 16] = [255, 0, 0]
    img[r + 6 : r + 10, c : c + 16] = [0, 0, 255]
    mask[r : r + 16, c : c + 16] = 255
    return img, mask


@pytest.fixture(scope="session")
def fake_mvtec(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create data/<category>/{train/good, test/good, test/scratch, ground_truth/scratch}."""
    root = tmp_path_factory.mktemp("mvtec")
    rng = np.random.default_rng(0)
    cat = root / "widget"
    for sub in ("train/good", "test/good", "test/scratch", "ground_truth/scratch"):
        (cat / sub).mkdir(parents=True)
    for i in range(12):
        Image.fromarray(good_image(rng)).save(cat / "train/good" / f"{i:03d}.png")
    for i in range(4):
        Image.fromarray(good_image(rng)).save(cat / "test/good" / f"{i:03d}.png")
        img, mask = defect_image(rng)
        Image.fromarray(img).save(cat / "test/scratch" / f"{i:03d}.png")
        Image.fromarray(mask).save(cat / "ground_truth/scratch" / f"{i:03d}_mask.png")
    return root


@pytest.fixture(scope="session")
def tiny_model(fake_mvtec: Path):
    """A PatchCore model with a random ResNet-18 fitted on the synthetic good images."""
    torch = pytest.importorskip("torch")
    from anomaly.data import list_train
    from anomaly.patchcore import assemble, build_embedder, fit_memory_bank, iter_batches

    torch.manual_seed(0)
    embedder = build_embedder(MODEL, PREP, pretrained=False)
    paths = list_train(fake_mvtec, "widget")
    bank, _ = fit_memory_bank(embedder, iter_batches(paths, 4, PREP), 0.5, 16, 0)
    return assemble(embedder, bank, MODEL, PREP.crop)


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory: pytest.TempPathFactory, tiny_model, fake_mvtec: Path) -> Path:
    """A models/ folder with an exported fp32 + int8 ONNX model and meta.json."""
    pytest.importorskip("onnx")
    from anomaly.export import export_onnx, quantize_int8

    root = tmp_path_factory.mktemp("models")
    cat_dir = root / "widget"
    export_onnx(tiny_model, cat_dir / "model_fp32.onnx", PREP.crop)
    from anomaly.data import list_train
    from anomaly.patchcore import iter_batches

    calib = list(iter_batches(list_train(fake_mvtec, "widget")[:6], 1, PREP))
    quantize_int8(cat_dir / "model_fp32.onnx", cat_dir / "model_int8.onnx", calib)
    meta = {
        "threshold": 1.0,
        "map_vmin": 0.0,
        "map_vmax": 2.0,
        "preprocess": {"resize": PREP.resize, "crop": PREP.crop},
    }
    (cat_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root
