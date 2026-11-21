"""Image preprocessing in numpy/PIL only.

Training, evaluation and the serving API all call these functions, so the
model sees identical pixels everywhere and the serving image needs no torch.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def open_image(source: Path | str | bytes) -> Image.Image:
    """Open an image from a path or raw bytes and convert it to RGB."""
    img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    return img.convert("RGB")


def resize_crop(img: Image.Image, resize: int, crop: int, resample: int) -> Image.Image:
    """Resize to a square of side `resize`, then center-crop to `crop`."""
    img = img.resize((resize, resize), resample=resample)
    offset = (resize - crop) // 2
    return img.crop((offset, offset, offset + crop, offset + crop))


def preprocess_image(img: Image.Image, resize: int = 256, crop: int = 224) -> np.ndarray:
    """Return a normalized float32 CHW array for the backbone."""
    img = resize_crop(img.convert("RGB"), resize, crop, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def display_image(img: Image.Image, resize: int = 256, crop: int = 224) -> np.ndarray:
    """Return the uint8 HWC RGB crop that the model actually sees (for overlays)."""
    return np.asarray(resize_crop(img.convert("RGB"), resize, crop, Image.Resampling.BILINEAR))


def load_mask(path: Path | None, resize: int = 256, crop: int = 224) -> np.ndarray:
    """Load a ground-truth mask as a binary uint8 array; all zeros if `path` is None."""
    if path is None:
        return np.zeros((crop, crop), dtype=np.uint8)
    mask = resize_crop(Image.open(path).convert("L"), resize, crop, Image.Resampling.NEAREST)
    return (np.asarray(mask) > 127).astype(np.uint8)
