#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Blender 3D Generator — macOS / Linux launcher
#  Place this file next to generate_3d.py, depth.png, texture.png
#
#  Usage:
#    ./run.sh [depth.png] [texture.png] [output.glb] [resolution] [depth_scale]
#
#  Defaults:
#    depth.png   texture.png   output.glb   512   0.5
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Auto-detect Blender ───────────────────────────────────────────────────────
BLENDER=""

# 1. PATH
if command -v blender &>/dev/null; then
    BLENDER="$(command -v blender)"
fi

# 2. macOS Application bundle
if [[ -z "$BLENDER" ]]; then
    for P in \
        "/Applications/Blender.app/Contents/MacOS/Blender" \
        "$HOME/Applications/Blender.app/Contents/MacOS/Blender"
    do
        [[ -x "$P" ]] && { BLENDER="$P"; break; }
    done
fi

# 3. Linux common paths
if [[ -z "$BLENDER" ]]; then
    for P in \
        "/usr/bin/blender" \
        "/usr/local/bin/blender" \
        "/snap/bin/blender" \
        "$HOME/.local/bin/blender"
    do
        [[ -x "$P" ]] && { BLENDER="$P"; break; }
    done
fi

if [[ -z "$BLENDER" ]]; then
    echo "[ERROR] Blender が見つかりません。"
    echo "        インストールしてください: https://www.blender.org/download/"
    echo "        または blender を PATH に追加してください。"
    exit 1
fi

echo "[INFO] Blender: $BLENDER"

# ── Arguments (with defaults) ─────────────────────────────────────────────────
DEPTH="${1:-depth.png}"
TEXTURE="${2:-texture.png}"
OUTPUT="${3:-output.glb}"
RESOLUTION="${4:-512}"
DEPTH_SCALE="${5:-0.5}"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[INFO] 生成開始..."
echo "       depth=$DEPTH  texture=$TEXTURE  output=$OUTPUT"
echo "       resolution=$RESOLUTION  depth_scale=$DEPTH_SCALE"

"$BLENDER" --background --python "$DIR/generate_3d.py" -- \
    --depth "$DIR/$DEPTH" \
    --texture "$DIR/$TEXTURE" \
    --output "$DIR/$OUTPUT" \
    --resolution "$RESOLUTION" \
    --depth_scale "$DEPTH_SCALE"

echo ""
echo "[OK] 完了: $OUTPUT"
echo "     gltf.report で確認: https://gltf.report/"
