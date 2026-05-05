"""
Depth Anything V2 via HuggingFace Inference API.
Tries Large → Small with retry on 503 (model cold-start).
"""
from __future__ import annotations

import io
import time

import numpy as np
import requests
from PIL import Image

_MODELS = [
    "depth-anything/Depth-Anything-V2-Small-hf",
    "depth-anything/Depth-Anything-V2-Large-hf",
    "Intel/dpt-hybrid-midas",
    "Intel/dpt-large",
]
_BASE = "https://api-inference.huggingface.co/models/"


def estimate_depth(image: Image.Image, hf_token: str) -> np.ndarray:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    img_bytes = buf.getvalue()
    headers = {"Authorization": f"Bearer {hf_token}"}

    last_err: Exception | None = None

    for model in _MODELS:
        for attempt in range(3):          # 503 はモデル起動待ちなので最大3回リトライ
            resp = requests.post(
                _BASE + model,
                headers=headers,
                data=img_bytes,
                timeout=90,
            )

            if resp.status_code == 200:
                depth_img = Image.open(io.BytesIO(resp.content)).convert("L")
                return np.array(depth_img, dtype=np.float32)

            if resp.status_code == 503:
                # モデル起動中 — 少し待ってリトライ
                wait = int(resp.headers.get("X-Wait-For-Model", "20"))
                time.sleep(min(wait, 30))
                continue

            # 401/403/422 など → 次のモデルへ
            last_err = requests.HTTPError(
                f"[{model}] HTTP {resp.status_code}: {resp.text[:200]}",
                response=resp,
            )
            break

    raise RuntimeError(
        "HuggingFace Inference API でエラーが発生しました。\n"
        "・HF Token が正しく設定されているか確認してください\n"
        "・しばらく待ってから再試行してください\n"
        f"最後のエラー: {last_err}"
    )
