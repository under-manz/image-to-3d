"""
Depth Anything V2 via HuggingFace Inference API (free with HF token).
Returns a much better depth map than grayscale for natural photos.
"""
from __future__ import annotations

import io

import numpy as np
import requests
from PIL import Image

_HF_API_URL = (
    "https://api-inference.huggingface.co/models/"
    "depth-anything/Depth-Anything-V2-Large"
)


def estimate_depth(image: Image.Image, hf_token: str) -> np.ndarray:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)

    headers = {"Authorization": f"Bearer {hf_token}"}
    resp = requests.post(
        _HF_API_URL,
        headers=headers,
        data=buf.getvalue(),
        timeout=90,
    )
    resp.raise_for_status()

    depth_img = Image.open(io.BytesIO(resp.content)).convert("L")
    return np.array(depth_img, dtype=np.float32)
