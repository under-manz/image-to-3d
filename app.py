import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
from PIL import Image

from generator import depth_estimator, mesh_builder, claude_analyzer
from generator import replicate_generator, onnx_depth


def _secret(key: str, user_input: str = "") -> str | None:
    if user_input:
        return user_input
    if os.environ.get(key):
        return os.environ[key]
    try:
        return st.secrets[key]
    except Exception:
        return None


st.set_page_config(page_title="Image → 3D (GLB)", layout="wide")
st.title("Image → 3D GLB Generator")
st.caption("PNG / JPEG をアップロードして GLB 形式の 3D モデルを生成します")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("生成モード")

    MODE_LABELS = {
        "hunyuan":    "★★★★  Hunyuan 3D（最高品質 / ~$0.05）",
        "triposr":    "★★★   TripoSR（高速高品質 / ~$0.01）",
        "midas_onnx": "★★    MiDaS ONNX（完全無料・API不要）",
        "grayscale":  "★     Grayscale（最速・低品質）",
    }

    mode = st.selectbox(
        "モード",
        list(MODE_LABELS.keys()),
        format_func=lambda k: MODE_LABELS[k],
        index=1,
    )

    st.divider()

    replicate_key = ""
    anthropic_key = ""

    if mode in ("hunyuan", "triposr"):
        replicate_key = st.text_input(
            "Replicate API Token",
            type="password",
            help="replicate.com/account/api-tokens",
        )
        if not _secret("REPLICATE_API_TOKEN", replicate_key):
            st.warning("Replicate API Token が必要です")

    anthropic_key = st.text_input(
        "Anthropic API Key（任意）",
        type="password",
    )

    if mode in ("midas_onnx", "grayscale"):
        st.divider()
        st.subheader("メッシュ設定")
        resolution  = st.select_slider("解像度", options=[64, 128, 256, 512], value=256)
        depth_scale = st.slider("奥行きスケール", 0.05, 1.0, 0.3, step=0.05)
        invert_depth = st.toggle("深度を反転", value=False)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg", "webp"])
if uploaded is None:
    st.info("PNG または JPEG をアップロードしてください")
    st.stop()

image = Image.open(uploaded).convert("RGB")
col_img, col_depth = st.columns(2)
col_img.image(image, caption="アップロード画像", use_container_width=True)

# ── Claude analysis ───────────────────────────────────────────────────────────
api_key = _secret("ANTHROPIC_API_KEY", anthropic_key)
if api_key:
    with st.spinner("Claude で画像を解析中..."):
        analysis = claude_analyzer.analyze(image, api_key)
    with st.expander("Claude 解析結果"):
        st.write(f"**説明:** {analysis['scene_description']}")
        if analysis.get("notes"):
            st.write(f"**注意:** {analysis['notes']}")

# ── Generate ──────────────────────────────────────────────────────────────────
if st.button("3D モデルを生成", type="primary", use_container_width=True):

    glb_bytes: bytes | None = None

    if mode == "hunyuan":
        token = _secret("REPLICATE_API_TOKEN", replicate_key)
        if not token:
            st.error("Replicate API Token を入力してください")
            st.stop()
        with st.spinner("Hunyuan 3D で生成中（1〜3分）..."):
            try:
                glb_bytes = replicate_generator.generate_hunyuan(image, token)
            except Exception as e:
                st.error(f"Hunyuan エラー: {e}")
                st.stop()

    elif mode == "triposr":
        token = _secret("REPLICATE_API_TOKEN", replicate_key)
        if not token:
            st.error("Replicate API Token を入力してください")
            st.stop()
        with st.spinner("TripoSR で生成中（30〜60秒）..."):
            try:
                glb_bytes = replicate_generator.generate_triposr(image, token)
            except Exception as e:
                st.error(f"TripoSR エラー: {e}")
                st.stop()

    elif mode == "midas_onnx":
        with st.spinner("MiDaS で深度推定中（初回はモデルDL約50MB）..."):
            try:
                import numpy as np
                depth = onnx_depth.estimate_depth(image)
            except Exception as e:
                st.error(f"深度推定エラー: {e}")
                st.stop()
        d_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype("uint8")
        col_depth.image(d_vis, caption="MiDaS 深度マップ", use_container_width=True)
        with st.spinner("メッシュ構築中..."):
            mesh = mesh_builder.build_mesh(depth, image, resolution, depth_scale, invert_depth)
            glb_bytes = mesh_builder.export_glb(mesh)

    else:  # grayscale
        import numpy as np
        depth = depth_estimator.estimate(image, method="grayscale")
        d_vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype("uint8")
        col_depth.image(d_vis, caption="Grayscale 深度マップ", use_container_width=True)
        with st.spinner("メッシュ構築中..."):
            mesh = mesh_builder.build_mesh(depth, image, resolution, depth_scale, invert_depth)
            glb_bytes = mesh_builder.export_glb(mesh)

    if glb_bytes:
        st.success(f"完了！  {len(glb_bytes)/1024:.1f} KB")
        stem = pathlib.Path(uploaded.name).stem
        st.download_button(
            label="GLB をダウンロード",
            data=glb_bytes,
            file_name=f"{stem}_3d.glb",
            mime="model/gltf-binary",
            use_container_width=True,
        )
        st.info("プレビュー: [gltf.report](https://gltf.report/) / [model-viewer](https://modelviewer.dev/editor)")
