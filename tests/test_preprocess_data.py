"""Tests for preprocessing and the MVTec folder layout helpers."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomaly.data import list_test, list_train, split_train_calib
from anomaly.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    display_image,
    load_mask,
    open_image,
    preprocess_image,
)


def test_preprocess_shape_dtype_and_normalization() -> None:
    img = Image.new("RGB", (300, 200), color=(124, 116, 104))
    x = preprocess_image(img, resize=256, crop=224)
    assert x.shape == (3, 224, 224) and x.dtype == np.float32
    expected = (np.array([124, 116, 104]) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(x[:, 100, 100], expected, atol=1e-5)


def test_open_image_from_bytes_converts_grayscale_to_rgb() -> None:
    buf = io.BytesIO()
    Image.new("L", (10, 10), 128).save(buf, format="PNG")
    assert open_image(buf.getvalue()).mode == "RGB"


def test_mask_is_binary_and_aligned_with_image_crop() -> None:
    mask = Image.new("L", (256, 256), 0)
    mask.paste(255, (0, 0, 128, 256))  # left half defective
    path_img = Image.new("RGB", (256, 256))
    path_img.paste((255, 255, 255), (0, 0, 128, 256))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    buf.seek(0)
    m = load_mask(buf, resize=256, crop=224)  # type: ignore[arg-type]
    assert set(np.unique(m)) <= {0, 1}
    rgb = display_image(path_img, 256, 224)
    np.testing.assert_array_equal(m == 1, rgb[..., 0] > 127)


def test_missing_mask_is_all_zeros() -> None:
    assert load_mask(None, 256, 224).sum() == 0


def test_list_train_and_test(fake_mvtec: Path) -> None:
    assert len(list_train(fake_mvtec, "widget")) == 12
    test = list_test(fake_mvtec, "widget")
    assert len(test) == 8
    defects = [s for s in test if s.label == 1]
    assert len(defects) == 4 and all(s.mask_path is not None for s in defects)
    assert all(s.mask_path is None for s in test if s.label == 0)
    assert list_train(fake_mvtec, "missing") == [] and list_test(fake_mvtec, "missing") == []


def test_split_is_deterministic_disjoint_and_complete() -> None:
    paths = [Path(f"{i:03d}.png") for i in range(50)]
    train, calib = split_train_calib(paths, 0.1, seed=0)
    assert len(calib) == 5 and len(train) == 45
    assert not set(train) & set(calib) and set(train) | set(calib) == set(paths)
    assert split_train_calib(paths, 0.1, seed=0) == (train, calib)
    assert split_train_calib(paths, 0.1, seed=1) != (train, calib)


def test_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError):
        split_train_calib([Path("a.png")], 1.0, 0)
