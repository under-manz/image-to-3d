"""
Convert a depth map + texture image into a trimesh.Trimesh and export GLB bytes.
"""
from __future__ import annotations

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual import TextureVisuals
from PIL import Image


def build_mesh(
    depth_map: np.ndarray,
    texture: Image.Image,
    resolution: int = 256,
    depth_scale: float = 0.3,
    invert_depth: bool = False,
) -> trimesh.Trimesh:
    """
    depth_map   : H×W float32 array (raw values from estimator)
    texture     : PIL Image used as colour texture
    resolution  : number of vertices per side (higher = more detail, heavier)
    depth_scale : Z range in model units (≈metres in GLB viewers)
    invert_depth: MiDaS returns inverse depth; set True to flip
    """
    # ── resize depth & texture to target resolution ─────────────────────────
    depth_img = Image.fromarray(depth_map).resize(
        (resolution, resolution), Image.BILINEAR
    )
    depth = np.array(depth_img, dtype=np.float32)

    tex_resized = texture.convert("RGBA").resize(
        (resolution, resolution), Image.LANCZOS
    )

    # ── normalise depth to [0, 1] ────────────────────────────────────────────
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    if invert_depth:
        depth = 1.0 - depth

    depth *= depth_scale

    # ── build vertex grid ────────────────────────────────────────────────────
    h = w = resolution
    xs = np.linspace(0.0, 1.0, w)
    ys = np.linspace(0.0, 1.0, h)
    xx, yy = np.meshgrid(xs, ys)          # both (h, w)

    vertices = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        depth.ravel(),
    ]).astype(np.float32)

    # ── build faces (vectorised) ─────────────────────────────────────────────
    i_idx = np.arange(h - 1)
    j_idx = np.arange(w - 1)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")
    base = (ii * w + jj).ravel()

    t1 = np.column_stack([base,       base + w,     base + 1    ])
    t2 = np.column_stack([base + 1,   base + w,     base + w + 1])
    faces = np.vstack([t1, t2]).astype(np.int32)

    # ── UV coordinates (v-axis flipped for GL convention) ────────────────────
    uv = np.column_stack([xx.ravel(), 1.0 - yy.ravel()]).astype(np.float32)

    # ── assemble trimesh with PBR texture ────────────────────────────────────
    material = PBRMaterial(
        baseColorTexture=tex_resized,
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    visual = TextureVisuals(uv=uv, material=material)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=visual,
        process=False,
    )
    return mesh


def export_glb(mesh: trimesh.Trimesh) -> bytes:
    return mesh.export(file_type="glb")
