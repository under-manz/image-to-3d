import os
import pathlib
import sys

# make generator importable regardless of cwd
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import streamlit as st
from PIL import Image

from generator import depth_estimator, mesh_builder, claude_analyzer

# ── Detect environment ───────────────────────────────────────────────────────
_MIDAS_AVAILABLE = False
try:
    import torch          # noqa: F401
    import transformers   # noqa: F401
    _MIDAS_AVAILABLE = True
except ImportError:
    pass

def _get_api_key(user_input: str) -> str | None:
    """Return API key from UI input → env var → st.secrets (in that order)."""
    if user_input:
        return user_input
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None

st.set_page_config(page_title="Image → 3D (GLB)", layout="wide")
st.title("Image → 3D GLB Generator")
st.caption("PNG / JPEG をアップロードして GLB 形式の 3D モデルを生成します")

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("設定")

    api_key_input = st.text_input(
        "Anthropic API Key（任意）",
        type="password",
        help="入力するとClaude Visionで画像を解析し、パラメータを自動提案します。"
             "クラウド版はサーバー側で設定済みの場合は不要です。",
    )
    api_key = _get_api_key(api_key_input)

    st.divider()
    st.subheader("深度推定")
    if _MIDAS_AVAILABLE:
        _method_options = ["auto", "midas", "grayscale"]
        _method_help = "auto: MiDaS を試みて失敗時 grayscale\nmidas: ML深度推定\ngrayscale: 輝度→高さ"
    else:
        _method_options = ["grayscale"]
        _method_help = "クラウド環境のため grayscale のみ利用可能（ローカルでは MiDaS も使用可）"
        st.caption("ℹ️ クラウド環境: grayscale モード固定")
    method = st.selectbox("手法", _method_options, help=_method_help)

    st.divider()
    st.subheader("メッシュ設定")
    resolution = st.select_slider(
        "解像度（頂点数/辺）",
        options=[64, 128, 256, 512],
        value=256,
        help="高いほど精細。512 は重くなります",
    )
    depth_scale = st.slider(
        "奥行きスケール",
        min_value=0.05,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Z 方向の強調倍率",
    )
    invert_depth = st.toggle(
        "深度を反転",
        value=False,
        help="暗い部分を手前にしたい場合にON",
    )

# ── Main ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "画像をアップロード",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded is None:
    st.info("PNG または JPEG をアップロードしてください")
    st.stop()

image = Image.open(uploaded).convert("RGB")
col_img, col_depth = st.columns(2)
col_img.image(image, caption="アップロード画像", use_container_width=True)

# ── Claude analysis ───────────────────────────────────────────────────────────
analysis = None
if api_key:
    with st.spinner("Claude で画像を解析中..."):
        analysis = claude_analyzer.analyze(image, api_key)

    with st.expander("Claude 解析結果", expanded=True):
        st.write(f"**説明:** {analysis['scene_description']}")
        if analysis.get("notes"):
            st.write(f"**注意:** {analysis['notes']}")

        apply = st.checkbox("解析結果をパラメータに反映する", value=True)
        if apply:
            method       = analysis.get("depth_method", method)
            depth_scale  = float(analysis.get("recommended_depth_scale", depth_scale))
            resolution   = int(analysis.get("recommended_resolution", resolution))
            invert_depth = bool(analysis.get("invert_depth", invert_depth))
            st.caption(
                f"反映: method={method}, depth_scale={depth_scale}, "
                f"resolution={resolution}, invert={invert_depth}"
            )

# ── Generate ──────────────────────────────────────────────────────────────────
if st.button("3D モデルを生成", type="primary", use_container_width=True):
    with st.spinner("深度マップを推定中..."):
        import numpy as np
        depth = depth_estimator.estimate(image, method=method)

    # Show depth map
    d_norm = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype("uint8")
    if invert_depth:
        d_norm = 255 - d_norm
    col_depth.image(d_norm, caption="深度マップ", use_container_width=True)

    with st.spinner(f"メッシュを構築中（解像度 {resolution}×{resolution}）..."):
        mesh = mesh_builder.build_mesh(
            depth_map=depth,
            texture=image,
            resolution=resolution,
            depth_scale=depth_scale,
            invert_depth=invert_depth,
        )

    with st.spinner("GLB を出力中..."):
        glb_bytes = mesh_builder.export_glb(mesh)

    st.success(
        f"完了！  頂点数: {len(mesh.vertices):,}  面数: {len(mesh.faces):,}  "
        f"サイズ: {len(glb_bytes)/1024:.1f} KB"
    )

    stem = pathlib.Path(uploaded.name).stem
    st.download_button(
        label="GLB をダウンロード",
        data=glb_bytes,
        file_name=f"{stem}_3d.glb",
        mime="model/gltf-binary",
        use_container_width=True,
    )

    st.info(
        "ダウンロードした .glb ファイルは "
        "[gltf.report](https://gltf.report/) や "
        "[model-viewer](https://modelviewer.dev/editor) でプレビューできます"
    )
