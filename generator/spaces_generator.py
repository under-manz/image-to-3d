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


def generate_stable_fast_3d(image: Image.Image) -> bytes:
    """
    stabilityai/stable-fast-3d — fast textured 3D mesh in <1 second.
    https://huggingface.co/spaces/stabilityai/stable-fast-3d
    """
    from gradio_client import Client, handle_file

    tmp = _save_tmp(image)
    try:
        client = Client("stabilityai/stable-fast-3d")
        result = client.predict(
            handle_file(tmp),
            0.5,          # foreground_ratio
            "none",       # background_choice: "none" / "grey" / "white"
            "triangle",   # remesh_choice
            0,            # vertex_count  (0 = auto)
            api_name="/run",
        )
        return _to_bytes(result)
    finally:
        os.unlink(tmp)


def _trellis_api_params(client) -> set[str]:
    """Discover available parameter names for /image_to_3d endpoint."""
    try:
        info = client.view_api(return_format="dict")
        for ep in info.get("named_endpoints", {}).values():
            if ep.get("parameters"):
                return {p["parameter_name"] for p in ep["parameters"]}
    except Exception:
        pass
    return set()


def generate_trellis2(
    image: Image.Image,
    extra_images: list[Image.Image] | None = None,
) -> bytes:
    """
    microsoft/TRELLIS.2 — high-quality structured 3D.
    extra_images: used if the Space supports multiimages, otherwise ignored.
    """
    from gradio_client import Client, handle_file

    tmp = _save_tmp(image)
    extra_tmps = [_save_tmp(img) for img in (extra_images or [])]
    try:
        client = Client("microsoft/TRELLIS.2")
        params = _trellis_api_params(client)

        kwargs: dict = {
            "image": handle_file(tmp),
            "seed": 0,
            "ss_guidance_strength": 7.5,
            "ss_sampling_steps": 12,
            "slat_guidance_strength": 3.0,
            "slat_sampling_steps": 12,
        }

        # multiimages は Space が対応している場合のみ追加
        if extra_tmps and "multiimages" in params:
            kwargs["multiimages"] = [handle_file(p) for p in extra_tmps]
            kwargs["multiimage_algo"] = "multidiffusion"
        elif "multiimages" in params:
            kwargs["multiimages"] = []
            kwargs["multiimage_algo"] = "stochastic"

        result = client.predict(**kwargs, api_name="/image_to_3d")
        state = result[0] if isinstance(result, (list, tuple)) else result

        glb_result = client.predict(
            state=state,
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb",
        )
        return _to_bytes(glb_result)
    finally:
        os.unlink(tmp)
        for p in extra_tmps:
            os.unlink(p)


def generate_triposg(image: Image.Image) -> bytes:
    """
    VAST-AI/TripoSG — high-fidelity 3D shape synthesis.
    https://huggingface.co/spaces/VAST-AI/TripoSG
    """
    from gradio_client import Client, handle_file

    tmp = _save_tmp(image)
    try:
        client = Client("VAST-AI/TripoSG")
        result = client.predict(
            handle_file(tmp),
            api_name="/generate",
        )
        return _to_bytes(result)
    finally:
        os.unlink(tmp)
