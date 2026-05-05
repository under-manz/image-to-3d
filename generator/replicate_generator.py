"""
High-quality image-to-3D via Replicate API.
"""
from __future__ import annotations

import os
import tempfile
import requests
from PIL import Image


def _save_tmp(image: Image.Image) -> str:
    """Save PIL image to a temp PNG file and return its path."""
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
        resp = requests.get(output, timeout=120)
        resp.raise_for_status()
        return resp.content

    if isinstance(output, list):
        for item in output:
            try:
                return _read_output(item)
            except Exception:
                continue

    if isinstance(output, dict):
        for key in ("glb", "mesh", "output", "url"):
            if key in output:
                return _read_output(output[key])

    raise ValueError(f"Cannot read Replicate output: {type(output)}")


def generate_triposr(image: Image.Image, api_token: str) -> bytes:
    import replicate

    os.environ["REPLICATE_API_TOKEN"] = api_token
    tmp_path = _save_tmp(image)
    try:
        with open(tmp_path, "rb") as f:
            output = replicate.run(
                "stability-ai/triposr",
                input={
                    "image": f,
                    "output_format": "glb",
                    "remove_background": True,
                    "foreground_ratio": 0.85,
                    "mc_resolution": 256,
                },
            )
    finally:
        os.unlink(tmp_path)

    return _read_output(output)


def generate_trellis(image: Image.Image, api_token: str) -> bytes:
    import replicate

    os.environ["REPLICATE_API_TOKEN"] = api_token
    tmp_path = _save_tmp(image)
    try:
        with open(tmp_path, "rb") as f:
            output = replicate.run(
                "zsxkib/trellis",
                input={
                    "image": f,
                    "output_format": "glb",
                },
            )
    finally:
        os.unlink(tmp_path)

    if isinstance(output, list):
        for item in output:
            try:
                raw = _read_output(item)
                if raw[:4] == b"glTF":
                    return raw
            except Exception:
                continue
        return _read_output(output[-1])

    return _read_output(output)
