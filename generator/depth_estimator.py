"""
Depth estimation from a PIL image.

Two backends are available:
  - MiDaS  : ML-based, requires torch + transformers (better quality)
  - Grayscale : pixel luminance as height (no ML deps, instant)
"""
from __future__ import annotations

import numpy as np
from PIL import Image


# ── MiDaS backend ──────────────────────────────────────────────────────────
def _estimate_midas(image: Image.Image) -> np.ndarray:
    from transformers import pipeline

    estimator = pipeline(
        task="depth-estimation",
        model="Intel/dpt-hybrid-midas",
    )
    result = estimator(image)
    depth = np.array(result["depth"], dtype=np.float32)
    return depth


# ── Grayscale fallback ──────────────────────────────────────────────────────
def _estimate_grayscale(image: Image.Image) -> np.ndarray:
    gray = image.convert("L")
    return np.array(gray, dtype=np.float32)


# ── Public API ──────────────────────────────────────────────────────────────
def estimate(image: Image.Image, method: str = "auto") -> np.ndarray:
    """
    Return a 2-D depth map (H × W, float32).
    Values are NOT normalised here — callers normalise as needed.

    method: "midas" | "grayscale" | "auto"
      auto → tries MiDaS first, falls back to grayscale on import error.
    """
    if method == "midas":
        return _estimate_midas(image)
    if method == "grayscale":
        return _estimate_grayscale(image)

    # auto
    try:
        return _estimate_midas(image)
    except Exception:
        return _estimate_grayscale(image)
