"""
Image-to-3D via public HuggingFace Spaces (Gradio API).
Completely free, no API key, no credit card required.
"""
from __future__ import annotations

import os
import tempfile
from PIL import Image


def _save_tmp(image: Image.Image) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.convert("RGB").save(tmp.name, format="PNG")
    tmp.close()
    return tmp.name


def _read_file(path) -> bytes:
    """Read GLB bytes from a local path returned by gradio_client."""
    if isinstance(path, tuple):
        path = path[0]
    if isinstance(path, dict):
        path = path.get("path") or path.get("name") or next(iter(path.values()))
    with open(str(path), "rb") as f:
        return f.read()


def generate_trellis(image: Image.Image) -> bytes:
    """
    JeffreyXiang/TRELLIS — high-quality structured 3D generation.
    Two-step: image_to_3d → extract_glb
    """
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

        # Step 2: state → GLB file
        glb_result = client.predict(
            state=state,
            mesh_simplify=0.95,
            texture_size=1024,
            api_name="/extract_glb",
        )
        glb_path = glb_result[0] if isinstance(glb_result, (list, tuple)) else glb_result
        return _read_file(glb_path)
    finally:
        os.unlink(tmp)


def generate_triposr(image: Image.Image) -> bytes:
    """
    stabilityai/TripoSR — fast single-image 3D reconstruction.
    """
    from gradio_client import Client, handle_file

    tmp = _save_tmp(image)
    try:
        client = Client("stabilityai/TripoSR")
        result = client.predict(
            handle_file(tmp),   # image
            True,               # do_remove_background
            0.9,                # foreground_ratio
            256,                # mc_resolution
            ["glb"],            # output_formats
            api_name="/generate",
        )
        # result: list of output file paths
        if isinstance(result, (list, tuple)):
            for item in result:
                try:
                    raw = _read_file(item)
                    if raw[:4] == b"glTF":
                        return raw
                except Exception:
                    continue
            return _read_file(result[-1])
        return _read_file(result)
    finally:
        os.unlink(tmp)


def generate_instantmesh(image: Image.Image) -> bytes:
    """
    TencentARC/InstantMesh — multi-view 3D reconstruction.
    """
    from gradio_client import Client, handle_file

    tmp = _save_tmp(image)
    try:
        client = Client("TencentARC/InstantMesh")
        # Step 1: preprocess
        preprocessed = client.predict(
            handle_file(tmp),
            True,   # remove_background
            api_name="/preprocess",
        )
        # Step 2: generate multi-view
        mv_result = client.predict(
            preprocessed,
            42,     # seed
            75,     # sample_steps
            api_name="/generate_mvs",
        )
        # Step 3: reconstruct 3D
        mesh_result = client.predict(
            mv_result,
            api_name="/make3d",
        )
        # mesh_result: (video_path, obj_path, glb_path)
        if isinstance(mesh_result, (list, tuple)):
            for item in reversed(mesh_result):
                try:
                    raw = _read_file(item)
                    if raw[:4] == b"glTF":
                        return raw
                except Exception:
                    continue
        return _read_file(mesh_result)
    finally:
        os.unlink(tmp)
