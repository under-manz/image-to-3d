"""
Image-to-3D via public HuggingFace Spaces (Gradio API).
Completely free, no API key, no credit card required.
"""
from __future__ import annotations

import os
import tempfile
import requests
from PIL import Image


def _save_tmp(image: Image.Image, size: int = 512) -> str:
    """Resize to square (white padding) and save as PNG."""
    img = image.convert("RGBA")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset, img)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    canvas.convert("RGB").save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


def _to_bytes(item) -> bytes:
    if hasattr(item, "path") and item.path:
        return _to_bytes(item.path)
    if hasattr(item, "url") and item.url:
        return _to_bytes(item.url)
    if isinstance(item, dict):
        for key in ("path", "name", "url", "value"):
            if item.get(key):
                return _to_bytes(item[key])
    if isinstance(item, str) and item.startswith("http"):
        resp = requests.get(item, timeout=180)
        resp.raise_for_status()
        return resp.content
    if isinstance(item, str):
        with open(item, "rb") as f:
            return f.read()
    if isinstance(item, (list, tuple)):
        candidates = []
        for sub in item:
            try:
                raw = _to_bytes(sub)
                candidates.append(raw)
                if raw[:4] == b"glTF":
                    return raw
            except Exception:
                continue
        if candidates:
            return candidates[-1]
    raise ValueError(f"Cannot convert to bytes: {type(item)}")


def _client(space_id: str, hf_token: str | None = None):
    from gradio_client import Client
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    return Client(space_id)


def generate_stable_fast_3d(image: Image.Image, hf_token: str | None = None) -> bytes:
    from gradio_client import handle_file
    tmp = _save_tmp(image)
    try:
        client = _client("stabilityai/stable-fast-3d", hf_token)
        result = client.predict(
            handle_file(tmp), 0.5, "none", "triangle", 0,
            api_name="/run",
        )
        return _to_bytes(result)
    finally:
        os.unlink(tmp)


def generate_trellis2(
    image: Image.Image,
    extra_images: list[Image.Image] | None = None,
    hf_token: str | None = None,
) -> bytes:
    from gradio_client import handle_file
    tmp = _save_tmp(image)
    extra_tmps = [_save_tmp(img) for img in (extra_images or [])]
    try:
        client = _client("microsoft/TRELLIS.2", hf_token)
        result = client.predict(handle_file(tmp), api_name="/image_to_3d")
        state = result[0] if isinstance(result, (list, tuple)) else result
        glb_result = client.predict(state, 0.95, 1024, api_name="/extract_glb")
        return _to_bytes(glb_result)
    finally:
        os.unlink(tmp)
        for p in extra_tmps:
            os.unlink(p)


def generate_triposg(image: Image.Image, hf_token: str | None = None) -> bytes:
    from gradio_client import handle_file
    tmp = _save_tmp(image)
    try:
        client = _client("VAST-AI/TripoSG", hf_token)
        result = client.predict(handle_file(tmp), api_name="/generate")
        return _to_bytes(result)
    finally:
        os.unlink(tmp)
