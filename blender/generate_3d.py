"""
Blender Python script: depth map + texture image → high-quality GLB

Usage (run from terminal):
  blender --background --python generate_3d.py -- ^
      --depth depth.png --texture texture.png --output output.glb ^
      [--resolution 512] [--depth_scale 1.0] [--smooth_levels 2]

Arguments after '--' are passed to this script.
"""

import sys
import argparse
import numpy as np

# ── Parse arguments ──────────────────────────────────────────────────────────
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--depth",        required=True,  help="Depth map PNG")
parser.add_argument("--texture",      required=True,  help="Texture image PNG")
parser.add_argument("--output",       required=True,  help="Output GLB path")
parser.add_argument("--resolution",   type=int,   default=512,
                    help="Mesh resolution per side (default: 512)")
parser.add_argument("--depth_scale",  type=float, default=0.5,
                    help="Z depth strength in Blender units (default: 0.5)")
parser.add_argument("--smooth_levels",type=int,   default=0,
                    help="Subdivision smooth passes after mesh creation (0=off)")
parser.add_argument("--invert_depth", action="store_true",
                    help="Invert depth map (bright = far)")
args = parser.parse_args(argv)

# ── Import bpy after arg parsing ─────────────────────────────────────────────
import bpy
import bmesh
from PIL import Image
from pathlib import Path

print(f"\n[generate_3d] depth={args.depth}")
print(f"[generate_3d] texture={args.texture}")
print(f"[generate_3d] output={args.output}")
print(f"[generate_3d] resolution={args.resolution}  depth_scale={args.depth_scale}\n")

# ── Clear scene ───────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=True)
for block in list(bpy.data.meshes) + list(bpy.data.materials) + list(bpy.data.images):
    bpy.data.batch_remove([block])

# ── Load & preprocess depth map ───────────────────────────────────────────────
res = args.resolution
depth_img = Image.open(args.depth).convert("L").resize((res, res), Image.BILINEAR)
depth = np.array(depth_img, dtype=np.float32) / 255.0
if args.invert_depth:
    depth = 1.0 - depth
depth *= args.depth_scale

h, w = depth.shape  # == res, res

# ── Build vertex grid (numpy → lists) ────────────────────────────────────────
xs = np.linspace(-1.0,  1.0, w)
ys = np.linspace( 1.0, -1.0, h)   # Y flipped for correct orientation
xx, yy = np.meshgrid(xs, ys)

verts = np.column_stack([xx.ravel(), yy.ravel(), depth.ravel()]).tolist()

# UV coordinates (u, v)
us = np.linspace(0.0, 1.0, w)
vs = np.linspace(1.0, 0.0, h)   # V flipped for GL convention
uu, vv = np.meshgrid(us, vs)
uvs = np.column_stack([uu.ravel(), vv.ravel()])

# Quad faces: each quad = [top-left, top-right, bottom-right, bottom-left]
i_idx = np.arange(h - 1)
j_idx = np.arange(w - 1)
ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")
base = (ii * w + jj).ravel()
faces = np.column_stack([base, base + 1, base + w + 1, base + w]).tolist()

print(f"[generate_3d] {len(verts):,} vertices  {len(faces):,} faces")

# ── Create Blender mesh ───────────────────────────────────────────────────────
mesh = bpy.data.meshes.new("DepthMesh")
mesh.from_pydata(verts, [], faces)
mesh.update()

# UV layer
uv_layer = mesh.uv_layers.new(name="UVMap")
for poly in mesh.polygons:
    for loop_idx, vert_idx in zip(poly.loop_indices, poly.vertices):
        uv_layer.data[loop_idx].uv = uvs[vert_idx]

# Smooth shading
for poly in mesh.polygons:
    poly.use_smooth = True

obj = bpy.data.objects.new("DepthMesh", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# ── Optional subdivision smooth ───────────────────────────────────────────────
if args.smooth_levels > 0:
    mod = obj.modifiers.new("Subdiv", "SUBSURF")
    mod.subdivision_type = "SIMPLE"
    mod.levels = args.smooth_levels
    mod.render_levels = args.smooth_levels
    bpy.ops.object.modifier_apply(modifier="Subdiv")

# Weighted normals for clean shading
mod_wn = obj.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
mod_wn.keep_sharp = False
bpy.ops.object.modifier_apply(modifier="WeightedNormal")

# ── Material: PBR + image texture ────────────────────────────────────────────
mat = bpy.data.materials.new("DepthMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links

for n in list(nodes):
    nodes.remove(n)

output_node   = nodes.new("ShaderNodeOutputMaterial")
principled    = nodes.new("ShaderNodeBsdfPrincipled")
tex_img_node  = nodes.new("ShaderNodeTexImage")
uv_node       = nodes.new("ShaderNodeTexCoord")

tex_image = bpy.data.images.load(str(Path(args.texture).resolve()))
tex_img_node.image = tex_image

principled.inputs["Roughness"].default_value  = 0.8
principled.inputs["Metallic"].default_value   = 0.0

links.new(uv_node.outputs["UV"],            tex_img_node.inputs["Vector"])
links.new(tex_img_node.outputs["Color"],    principled.inputs["Base Color"])
links.new(principled.outputs["BSDF"],       output_node.inputs["Surface"])

output_node.location  = (400, 0)
principled.location   = (200, 0)
tex_img_node.location = (-100, 0)
uv_node.location      = (-400, 0)

mesh.materials.append(mat)

# ── Export GLB ────────────────────────────────────────────────────────────────
out = str(Path(args.output).resolve())
bpy.ops.export_scene.gltf(
    filepath=out,
    export_format="GLB",
    use_selection=False,
    export_materials="EXPORT",
    export_image_format="AUTO",
    export_texcoords=True,
    export_normals=True,
)

print(f"\n[generate_3d] ✓ Exported: {out}\n")
