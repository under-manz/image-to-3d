import os
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
from PIL import Image

from generator import depth_estimator, mesh_builder, claude_analyzer
from generator import spaces_generator, onnx_depth, bg_remover


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
        st.caption("HuggingFace Spaces 使用（ZeroGPU）")
        st.caption("⚠️ HF Token が必要です（無料アカウントで取得可）")

    st.divider()
    hf_key = st.text_input(
        "HuggingFace Token（Spaces使用時に必要）",
        type="password",
        help="huggingface.co/settings/tokens で Write トークンを取得",
    )

    st.divider()
    remove_bg = st.toggle(
        "背景を除去する",
        value=True,
        help="ONにすると3D生成前に背景を自動除去します。壁が生成されるのを防ぎます。",
    )

    st.divider()
    anthropic_key = st.text_input("Anthropic API Key（任意）", type="password")

    if mode in ("midas_onnx", "grayscale"):
        st.divider()
        st.subheader("メッシュ設定")
        resolution   = st.select_slider("解像度", options=[64, 128, 256, 512], value=256)
        depth_scale  = st.slider("奥行きスケール", 0.05, 1.0, 0.3, step=0.05)
        invert_depth = st.toggle("深度を反転", value=False)

# ── Upload ────────────────────────────────────────────────────────────────────
if mode == "trellis2":
    st.caption("💡 前面は必須。背面・側面を追加すると精度が上がります。")
    view_labels = ["前面（メイン・必須）", "背面（任意）", "左側面（任意）", "右側面（任意）"]
    cols = st.columns(4)
    uploaded_files = []
    for col, label in zip(cols, view_labels):
        f = col.file_uploader(label, type=["png", "jpg", "jpeg", "webp"], key=label)
        if f:
            uploaded_files.append(f)
else:
    f = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg", "webp"])
    uploaded_files = [f] if f else []

if not uploaded_files:
    st.info("PNG または JPEG をアップロードしてください")
    st.stop()

images = [Image.open(f).convert("RGB") for f in uploaded_files]
image = images[0]
extra_images = images[1:]

if mode == "trellis2":
    st.caption(f"アップロード済み: {len(images)} 枚")

col_img, col_depth = st.columns(2)
col_img.image(image, caption=f"メイン画像（+ {len(extra_images)}枚）" if extra_images else "アップロード画像", use_container_width=True)

# ── Claude analysis ───────────────────────────────────────────────────────────
api_key = _secret("ANTHROPIC_API_KEY", anthropic_key)
if api_key:
    with st.spinner("Claude で画像を解析中..."):
        analysis = claude_analyzer.analyze(image, api_key)
    with st.expander("Claude 解析結果"):
        st.write(f"**説明:** {analysis['scene_description']}")
        if analysis.get("notes"):
            st.write(f"**注意:** {analysis['notes']}")

# ── Background removal preview ───────────────────────────────────────────────
process_image = image
if remove_bg:
    with st.spinner("背景を除去中..."):
        try:
            rgba = bg_remover.remove_background(image)
            process_image = bg_remover.to_white_bg(rgba)
            col_depth.image(rgba, caption="背景除去後", use_container_width=True)
        except Exception as e:
            st.warning(f"背景除去をスキップしました: {e}")

# ── Generate ──────────────────────────────────────────────────────────────────
if st.button("3D モデルを生成", type="primary", use_container_width=True):

    glb_bytes: bytes | None = None

    hf_token = _secret("HF_TOKEN", hf_key) or None

    if mode == "trellis2":
        n = len(images)
        msg = f"TRELLIS.2 で生成中（{n}枚使用 / 2〜5分）..."
        with st.spinner(msg):
            try:
                extra = [bg_remover.to_white_bg(bg_remover.remove_background(img))
                         for img in extra_images] if remove_bg else extra_images
                glb_bytes = spaces_generator.generate_trellis2(process_image, extra, hf_token)
            except Exception as e:
                st.error(f"エラー: {e}")
                st.code(traceback.format_exc())
                st.stop()

    elif mode in ("sf3d", "triposg"):
        SPACE_TASKS = {
            "sf3d":    ("Stable Fast 3D で生成中（30秒〜2分）...", spaces_generator.generate_stable_fast_3d),
            "triposg": ("TripoSG で生成中（1〜3分）...",           spaces_generator.generate_triposg),
        }
        spinner_msg, fn = SPACE_TASKS[mode]
        with st.spinner(spinner_msg):
            try:
                glb_bytes = fn(process_image, hf_token)
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
