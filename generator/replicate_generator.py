"""
High-quality image-to-3D via Replicate API.
"""
from __future__ import annotations

import os
import tempfile
import time
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


def _run_with_retry(api_token: str, model: str, input_data: dict, retries: int = 3) -> bytes:
    import replicate
    os.environ["REPLICATE_API_TOKEN"] = api_token

    for attempt in range(retries):
        try:
            output = replicate.run(model, input=input_data)
            return _read_output(output)
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < retries - 1:
                # レート制限 → 少し待ってリトライ
                time.sleep(5 * (attempt + 1))
                continue
            raise


def generate_triposr(image: Image.Image, api_token: str) -> bytes:
    tmp = _save_tmp(image)
    try:
        with open(tmp, "rb") as f:
            return _run_with_retry(
                api_token,
                "camenduru/tripo-sr:e0d3fe8abce3ba86497ea3530d9eae59af7b2231b6c82bedfc32b0732d35ec3a",
                {
                    "image": f,
                    "do_remove_background": True,
                    "foreground_ratio": 0.9,
                    "marching_cubes_resolution": 256,
                },
            )
    finally:
        os.unlink(tmp)


def generate_hunyuan(image: Image.Image, api_token: str) -> bytes:
    tmp = _save_tmp(image)
    try:
        with open(tmp, "rb") as f:
            return _run_with_retry(
                api_token,
                "tencent/hunyuan-3d-3.1",
                {
                    "image": f,
                    "enable_pbr": True,
                },
            )
    finally:
        os.unlink(tmp)
