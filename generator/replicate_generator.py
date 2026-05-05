"""
High-quality image-to-3D via Replicate API.

Models:
  trellis  : firtoz/trellis          — ~$0.034/run, ~25秒
  triposr  : camenduru/tripo-sr      — 高速・軽量
  hunyuan  : tencent/hunyuan-3d-3.1  — 高品質テクスチャ
"""
from __future__ import annotations

import os
import tempfile
import requests
from PIL import Image


def _save_tmp(image: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.convert("RGB").save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


def _read_output(output) -> bytes:
    if output is None:
        raise ValueError("Replicate returned no output")

    if hasattr(output, "read"):
        data = output.read()
        return data if isinstance(data, bytes) else bytes(data)

    if isinstance(output, str) and output.startswith("http"):
        resp = requests.get(output, timeout=180)
        resp.raise_for_status()
        return resp.content

    if isinstance(output, list):
        # GLBマジックバイト b"glTF" を優先
        results = []
        for item in output:
            try:
                raw = _read_output(item)
                results.append(raw)
                if raw[:4] == b"glTF":
                    return raw
            except Exception:
                continue
        if results:
            return results[-1]

    if isinstance(output, dict):
        for key in ("glb", "mesh", "model", "output", "url"):
            if key in output:
                return _read_output(output[key])

    raise ValueError(f"Cannot read Replicate output: {type(output)}")


def _run(api_token: str, model: str, **input_kwargs) -> bytes:
    import replicate
    os.environ["REPLICATE_API_TOKEN"] = api_token
    output = replicate.run(model, input=input_kwargs)
    return _read_output(output)


def generate_trellis(image: Image.Image, api_token: str) -> bytes:
    tmp = _save_tmp(image)
    try:
        with open(tmp, "rb") as f:
            return _run(api_token, "firtoz/trellis", image=f)
    finally:
        os.unlink(tmp)


def generate_triposr(image: Image.Image, api_token: str) -> bytes:
    tmp = _save_tmp(image)
    try:
        with open(tmp, "rb") as f:
            return _run(api_token, "camenduru/tripo-sr", image=f)
    finally:
        os.unlink(tmp)


def generate_hunyuan(image: Image.Image, api_token: str) -> bytes:
    tmp = _save_tmp(image)
    try:
        with open(tmp, "rb") as f:
            return _run(
                api_token,
                "tencent/hunyuan-3d-3.1",
                image=f,
                export_format="glb",
            )
    finally:
        os.unlink(tmp)
