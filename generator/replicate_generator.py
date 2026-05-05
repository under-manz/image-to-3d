"""
High-quality image-to-3D via Replicate API.

Models:
  triposr  : stability-ai/triposr  — fast, good quality
  trellis  : zsxkib/trellis        — slower, best quality
"""
from __future__ import annotations

import io
import os
import requests
from PIL import Image


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def _read_output(output) -> bytes:
    """Normalise various Replicate output types to raw bytes."""
    if output is None:
        raise ValueError("Replicate returned no output")

    # replicate >= 0.25: FileOutput with .read()
    if hasattr(output, "read"):
        data = output.read()
        if isinstance(data, bytes):
            return data
        # some SDKs return a coroutine — shouldn't happen in sync mode but guard anyway
        return bytes(data)

    # plain URL string
    if isinstance(output, str) and output.startswith("http"):
        resp = requests.get(output, timeout=120)
        resp.raise_for_status()
        return resp.content

    # list of outputs — return first readable item
    if isinstance(output, list):
        for item in output:
            try:
                return _read_output(item)
            except Exception:
                continue

    # dict with a url/glb key
    if isinstance(output, dict):
        for key in ("glb", "mesh", "output", "url"):
            if key in output:
                return _read_output(output[key])

    raise ValueError(f"Cannot read Replicate output of type {type(output)}")


def generate_triposr(image: Image.Image, api_token: str) -> bytes:
    """
    stability-ai/triposr: fast single-image → textured GLB.
    Docs: https://replicate.com/stability-ai/triposr
    """
    import replicate

    os.environ["REPLICATE_API_TOKEN"] = api_token
    img_bytes = _image_to_png_bytes(image)

    output = replicate.run(
        "stability-ai/triposr",
        input={
            "image": io.BytesIO(img_bytes),
            "output_format": "glb",
            "remove_background": True,
            "foreground_ratio": 0.85,
            "mc_resolution": 256,
        },
    )
    return _read_output(output)


def generate_trellis(image: Image.Image, api_token: str) -> bytes:
    """
    zsxkib/trellis: structured 3D latent diffusion → high-quality GLB.
    Docs: https://replicate.com/zsxkib/trellis
    """
    import replicate

    os.environ["REPLICATE_API_TOKEN"] = api_token
    img_bytes = _image_to_png_bytes(image)

    output = replicate.run(
        "zsxkib/trellis",
        input={
            "image": io.BytesIO(img_bytes),
            "output_format": "glb",
        },
    )

    # TRELLIS may return a list; prefer actual GLB bytes (magic: b"glTF")
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
