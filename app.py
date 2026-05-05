import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
from PIL import Image

from generator import depth_estimator, mesh_builder, claude_analyzer
from generator import replicate_generator, hf_depth

# ── local MiDaS availability ─────────────────────────────────────────────────
_MIDAS_AVAILABLE = False
try:
    import torch        # noqa: F401
    import transformers # noqa: F401
    _MIDAS_AVAILABLE = True
except ImportError:
    pass


def _secret(key: str, user_input: str = "") -> str | None:
    if user_input:
        return user_input
    if os.environ.get(key):
        return os.environ[key]
    try:
        return st.secrets[key]
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Image → 3D (GLB)", layout="wide")
st.title("Image → 3D GLB Generator")
st.caption("PNG / JPEG をアップロードして GLB 形式の 3D モデルを生成します")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("生成モード")

    MODE_LABELS = {
        "trellis":   "★★★★  TRELLIS（最高品質 / Replicate）",
        "triposr":   "★★★   TripoSR（高品質 / Replicate）",
        "hf_depth":  "★★    Depth Anything V2（HuggingFace）",
        "grayscale": "★     Grayscale（API不要・低品質）",
    }
    if _MIDAS_AVAILABLE:
        MODE_LABELS["midas"] = "★★    MiDaS（ローカルのみ）"

    mode = st.selectbox(
        "モード",
        list(MODE_LABELS.keys()),
        format_func=lambda k: MODE_LABELS[k],
    )

    st.divider()
    st.subheader("API キー")

    replicate_key = ""
    hf_key = ""
    anthropic_key = ""

    if mode in ("trellis", "triposr"):
        replicate_key = st.text_input(
            "Replicate API Token",
            type="password",
            help="https://replicate.com/account/api-tokens",
        )
        if not _secret("REPLICATE_API_TOKEN", replicate_key):
            st.warning("Replicate API Token が必要です")

    if mode == "hf_depth":
        hf_key = st.text_input(
            "HuggingFace Token",
            type="password",
            help="https://huggingface.co/settings/tokens （無料）",
        )
        if not _secret("HF_TOKEN", hf_key):
            st.warning("HuggingFace Token が必要です")

    anthropic_key = st.text_input(
        "Anthropic API Key（任意 / Claude解析用）",
        type="password",
    )

    if mode in ("hf_depth", "grayscale", "midas"):
        st.divider()
        st.subheader("メッシュ設定")
        resolution = st.select_slider(
            "解像度（頂点数/辺）", options=[64, 128, 256, 512], value=256
        )
        depth_scale = st.slider("奥行きスケール", 0.05, 1.0, 0.3, step=0.05)
        invert_depth = st.toggle("深度を反転", value=False)

# ── Image upload ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg", "webp"])
if uploaded is None:
    st.info("PNG または JPEG をアップロードしてください")
    st.stop()

image = Image.open(uploaded).convert("RGB")

col_img, col_depth = st.columns(2)
col_img.image(image, caption="アップロード画像", use_container_width=True)

# ── Claude analysis (optional) ───────────────────────────────────────────────
api_key = _secret("ANTHROPIC_API_KEY", anthropic_key)
if api_key:
    with st.spinner("Claude で画像を解析中..."):
        analysis = claude_analyzer.analyze(image, api_key)
    with st.expander("Claude 解析結果"):
        st.write(f"**説明:** {analysis['scene_description']}")
        if analysis.get("notes"):
            st.write(f"**注意:** {analysis['notes']}")

# ── Generate ─────────────────────────────────────────────────────────────────
if st.button("3D モデルを生成", type="primary", use_container_width=True):

    glb_bytes: bytes | None = None

    # ── Replicate: TRELLIS ──────────────────────────────────────────────────
    if mode == "trellis":
        token = _secret("REPLICATE_API_TOKEN", replicate_key)
        if not token:
            st.error("Replicate API Token を入力してください")
            st.stop()
        with st.spinner("TRELLIS で 3D 生成中（1〜3分）..."):
            glb_bytes = replicate_generator.generate_trellis(image, token)

    # ── Replicate: TripoSR ──────────────────────────────────────────────────
    elif mode == "triposr":
        token = _secret("REPLICATE_API_TOKEN", replicate_key)
        if not token:
            st.error("Replicate API Token を入力してください")
            st.stop()
        with st.spinner("TripoSR で 3D 生成中（30〜60秒）..."):
            glb_bytes = replicate_generator.generate_triposr(image, token)

    # ── HuggingFace: Depth Anything V2 + mesh ──────────────────────────────
    elif mode == "hf_depth":
        token = _secret("HF_TOKEN", hf_key)
        if not token:
            st.error("HuggingFace Token を入力してください")
            st.stop()
        with st.spinner("Depth Anything V2 で深度推定中（モデル初回起動時は1分ほどかかります）..."):
            try:
                depth = hf_depth.estimate_depth(image, token)
            except Exception as e:
                st.error(str(e))
                st.stop()
        d_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype("uint8")
        col_depth.image(d_vis, caption="Depth Anything V2 深度マップ", use_container_width=True)
        with st.spinner("メッシュ構築中..."):
            mesh = mesh_builder.build_mesh(depth, image, resolution, depth_scale, invert_depth)
            glb_bytes = mesh_builder.export_glb(mesh)

    # ── MiDaS (local only) ──────────────────────────────────────────────────
    elif mode == "midas":
        with st.spinner("MiDaS で深度推定中..."):
            import numpy as np
            depth = depth_estimator.estimate(image, method="midas")
        d_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype("uint8")
        col_depth.image(d_vis, caption="MiDaS 深度マップ", use_container_width=True)
        with st.spinner("メッシュ構築中..."):
            mesh = mesh_builder.build_mesh(depth, image, resolution, depth_scale, invert_depth)
            glb_bytes = mesh_builder.export_glb(mesh)

    # ── Grayscale fallback ──────────────────────────────────────────────────
    else:
        import numpy as np
        depth = depth_estimator.estimate(image, method="grayscale")
        d_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype("uint8")
        col_depth.image(d_vis, caption="Grayscale 深度マップ", use_container_width=True)
        with st.spinner("メッシュ構築中..."):
            mesh = mesh_builder.build_mesh(depth, image, resolution, depth_scale, invert_depth)
            glb_bytes = mesh_builder.export_glb(mesh)

    # ── Download ─────────────────────────────────────────────────────────────
    if glb_bytes:
        size_kb = len(glb_bytes) / 1024
        st.success(f"完了！  GLB サイズ: {size_kb:.1f} KB")

        stem = pathlib.Path(uploaded.name).stem
        st.download_button(
            label="GLB をダウンロード",
            data=glb_bytes,
            file_name=f"{stem}_3d.glb",
            mime="model/gltf-binary",
            use_container_width=True,
        )
        st.info(
            "プレビュー: [gltf.report](https://gltf.report/)  /  "
            "[model-viewer](https://modelviewer.dev/editor)"
        )
