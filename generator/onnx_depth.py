"""
MiDaS-small depth estimation via ONNX Runtime (CPU).
No API key, no account required — model is downloaded once to /tmp.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
from PIL import Image

_MODEL_URL  = "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx"
_MODEL_PATH = "/tmp/midas_small.onnx"
_INPUT_SIZE = 256


def _download_model() -> None:
    if not os.path.exists(_MODEL_PATH):
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


def estimate_depth(image: Image.Image) -> np.ndarray:
    import onnxruntime as ort

    _download_model()

    img = image.convert("RGB").resize((_INPUT_SIZE, _INPUT_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    arr = arr.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    session = ort.InferenceSession(_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: arr})[0]

    depth = result[0]
    if depth.ndim == 3:
        depth = depth[0]
    return depth.astype(np.float32)
