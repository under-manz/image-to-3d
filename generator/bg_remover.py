"""
Background removal using rembg (u2netp — lightweight ~5MB model).
Returns RGBA PIL Image with background set to transparent.
"""
from __future__ import annotations

from PIL import Image


def remove_background(image: Image.Image) -> Image.Image:
    from rembg import remove, new_session
    session = new_session("u2netp")
    rgba = remove(image.convert("RGBA"), session=session)
    return rgba


def to_white_bg(rgba: Image.Image) -> Image.Image:
    """Composite RGBA onto white for models that don't support transparency."""
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg
