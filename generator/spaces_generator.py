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
    """Resize to square (padding with white) and save as PNG."""
    img = image.convert("RGBA")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset, img)
    result = canvas.convert("RGB")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    result.save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


def _to_bytes(item) -> bytes:
    """
    Convert any gradio_client output item to raw bytes.
    Handles: local path str, FileData object, dict, URL str, list/tuple.
    """
    # FileData object (gradio_client >= 1.0)
    if hasattr(item, "path") and item.path:
        return _to_bytes(item.path)
    if hasattr(item, "url") and item.url:
        return _to_bytes(item.url)

    # dict
    if isinstance(item, dict):
        for key in ("path", "name", "url", "value"):
            if item.get(key):
                return _to_bytes(item[key])

    # URL
    if isinstance(item, str) and item.startswith("http"):
        resp = requests.get(item, timeout=180)
        resp.raise_for_status()
        return resp.content

    # local path
    if isinstance(item, str):
        if not os.path.exists(item):
            raise FileNotFoundError(f"Output file not found: {item}")
        with open(item, "rb") as f:
            return f.read()

    # list / tuple — prefer GLB magic bytes
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

    raise ValueError(f"Cannot convert to bytes: {type(item)} = {repr(item)[:200]}")


def generate_trellis(image: Image.Image) -> bytes:
    from gradio_client import Client, handle_file

    tmp = _save_tmp(image)
    try:
        client = Client("JeffreyXiang/TRELLIS")

        # Step 1: image → 3D state
        result = client.predict(
            image=handle_file(tmp),
            multiimages=[],
            seed=0,
            ss_guidance_strength=7.5,
            ss_sampling_steps=12,
            slat_guidance_strength=3.0,
            slat_sampling_steps=12,
            multiimage_algo="stochastic",
            api_name="/image_to_3d",
        )
        state = result[0] if isinstance(result, (list, tuple)) else result

        # Step 2: state → GLB
        glb_result = client.predict(
            state=state,
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb",
        )
        return _to_bytes(glb_result)
    finally:
        os.unlink(tmp)


def generate_triposr(image: Image.Image) -> bytes:
    from gradio_client import Client, handle_file

    tmp = _save_tmp(image)
    try:
        client = Client("stabilityai/TripoSR")
        result = client.predict(
            handle_file(tmp),
            True,
            0.9,
            256,
            ["glb"],
            api_name="/generate",
        )
        return _to_bytes(result)
    finally:
        os.unlink(tmp)


def generate_instantmesh(image: Image.Image) -> bytes:
    from gradio_client import Client, handle_file

    tmp = _save_tmp(image)
    try:
        client = Client("TencentARC/InstantMesh")

        preprocessed = client.predict(
            handle_file(tmp),
            True,
            api_name="/preprocess",
        )
        mv_result = client.predict(
            preprocessed,
            42,
            75,
            api_name="/generate_mvs",
        )
        mesh_result = client.predict(
            mv_result,
            api_name="/make3d",
        )
        return _to_bytes(mesh_result)
    finally:
        os.unlink(tmp)
