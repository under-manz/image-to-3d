"""
Send an image to Claude Vision and get 3D-generation hints back.
"""
from __future__ import annotations

import base64
import json
import re
from io import BytesIO

import anthropic
from PIL import Image


_PROMPT = """\
この画像を3Dメッシュ生成のために分析してください。
以下のキーを持つJSONのみを返してください（説明文は不要）:

{
  "scene_description": "シーンの日本語説明（1〜2文）",
  "depth_method": "midas または grayscale の推奨",
  "invert_depth": true または false,
  "recommended_depth_scale": 0.05〜1.0 の数値,
  "recommended_resolution": 128 か 256 か 512,
  "notes": "3D化に際して注意すべき点（任意）"
}

depth_method の選び方:
  - 写真・自然画像・風景 → "midas"
  - ハイトマップ・地形図・白黒グラデーション → "grayscale"

invert_depth の選び方:
  - 明るいほど手前にある画像 → false
  - 暗いほど手前にある画像 → true
"""


def analyze(image: Image.Image, api_key: str | None = None) -> dict:
    """
    Returns a dict with keys defined in _PROMPT.
    Falls back to safe defaults on any error.
    """
    defaults = {
        "scene_description": "（Claude解析スキップ）",
        "depth_method": "auto",
        "invert_depth": False,
        "recommended_depth_scale": 0.3,
        "recommended_resolution": 256,
        "notes": "",
    }

    try:
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

        buf = BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=85)
        b64 = base64.standard_b64encode(buf.getvalue()).decode()

        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
        )

        raw = resp.content[0].text
        # Extract JSON even if model wraps it in markdown fences
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            defaults.update(parsed)
    except Exception:
        pass  # return defaults on any failure

    return defaults
