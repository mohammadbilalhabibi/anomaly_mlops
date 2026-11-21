"""Render anomaly maps as coloured heatmaps (numpy + PIL only, no matplotlib)."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# Anchor colours of a perceptually ordered dark-blue -> yellow ramp (similar to "inferno").
_ANCHORS = np.array(
    [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
     [245, 125, 21], [250, 193, 39], [252, 255, 164]],
    dtype=np.float32,
)  # fmt: skip


def _lut() -> np.ndarray:
    """256-entry RGB lookup table interpolated from the anchors."""
    pos = np.linspace(0, 1, len(_ANCHORS))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, pos, _ANCHORS[:, c]) for c in range(3)], axis=1).astype(np.uint8)


LUT = _lut()


def normalize_map(amap: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Scale an anomaly map to [0, 1] with fixed bounds (comparable across images)."""
    span = max(vmax - vmin, 1e-12)
    return np.clip((amap - vmin) / span, 0.0, 1.0)


def colorize(norm: np.ndarray) -> np.ndarray:
    """Map a [0, 1] array to uint8 RGB with the lookup table."""
    return LUT[np.round(norm * 255).astype(np.uint8)]


def overlay(rgb: np.ndarray, norm: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend the heatmap over the image; low scores stay mostly transparent."""
    heat = colorize(norm).astype(np.float32)
    a = (alpha * np.clip(norm * 1.5, 0.0, 1.0))[..., None]
    out = rgb.astype(np.float32) * (1 - a) + heat * a
    return out.clip(0, 255).astype(np.uint8)


def to_png_base64(rgb: np.ndarray) -> str:
    """Encode a uint8 RGB array as a base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
