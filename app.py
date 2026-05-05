import os
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
from PIL import Image

from generator import depth_estimator, mesh_builder, claude_analyzer
from generator import spaces_generator, onnx_depth


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

with st.sidebar:
    st.header("生成モード")

    MODE_LABELS = {
        "sf3d":       "★★★★  Stable Fast 3D（高速・無料）",
        "trellis2":   "★★★★  TRELLIS.2 Microsoft（最高品質・無料）",
        "triposg":    "★★★   TripoSG（高品質・無料）",
        "midas_onnx": "★★    MiDaS ONNX（ローカル・無料）",
        "grayscale":  "★     Grayscale（最速・低品質）",
    }

    mode = st.selectbox(
        "モード",
        list(MODE_LABELS.keys()),
        format_func=lambda k: MODE_LABELS[k],
        index=0,
    )

    if mode in ("sf3d", "trellis2", "triposg"):
        st.caption("HuggingFace Spaces 使用（カード不要・アカウント不要）")
        st.caption("⚠️ Space が混雑・休止中の場合は数分かかることがあります")

    st.divider()
    anthropic_key = st.text_input("Anthropic API Key（任意）", type="password")

    if mode in ("midas_onnx", "grayscale"):
        st.divider()
        st.subheader("メッシュ設定")
        resolution   = st.select_slider("解像度", options=[64, 128, 256, 512], value=256)
        depth_scale  = st.slider("奥行きスケール", 0.05, 1.0, 0.3, step=0.05)
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

    SPACE_TASKS = {
        "sf3d":     ("Stable Fast 3D で生成中（30秒〜2分）...",    spaces_generator.generate_stable_fast_3d),
        "trellis2": ("TRELLIS.2 で生成中（2〜5分）...",            spaces_generator.generate_trellis2),
        "triposg":  ("TripoSG で生成中（1〜3分）...",              spaces_generator.generate_triposg),
    }

    if mode in SPACE_TASKS:
        spinner_msg, fn = SPACE_TASKS[mode]
        with st.spinner(spinner_msg):
            try:
                glb_bytes = fn(image)
            except Exception as e:
                st.error(f"エラー: {e}")
                st.code(traceback.format_exc())
                st.stop()

    elif mode == "midas_onnx":
        with st.spinner("MiDaS で深度推定中（初回はモデルDL ~50MB）..."):
            try:
                import numpy as np
                depth = onnx_depth.estimate_depth(image)
            except Exception as e:
                st.error(f"深度推定エラー: {e}")
                st.code(traceback.format_exc())
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
