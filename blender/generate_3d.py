"""
Blender Python script: depth map + texture image -> solid GLB
- Front face: depth-displaced grid
- Side walls + back plate: closed solid so it looks 3D from all angles
- PIL-free: uses bpy image loading + numpy only
"""

import sys
import argparse
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--depth",         required=True)
parser.add_argument("--texture",       required=True)
parser.add_argument("--output",        required=True)
parser.add_argument("--resolution",    type=int,   default=256)
parser.add_argument("--depth_scale",   type=float, default=0.4)
parser.add_argument("--smooth_levels", type=int,   default=1)
parser.add_argument("--invert_depth",  action="store_true")
args = parser.parse_args(argv)

import bpy
import bmesh
from pathlib import Path

print(f"\n[generate_3d] depth={args.depth}  texture={args.texture}")
print(f"[generate_3d] resolution={args.resolution}  depth_scale={args.depth_scale}\n")

# ── Clear scene ───────────────────────────────────────────────────────────────
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=True)
for block in list(bpy.data.meshes) + list(bpy.data.materials) + list(bpy.data.images):
    bpy.data.batch_remove([block])

res = args.resolution

# ── Load depth map (bpy only, no PIL) ────────────────────────────────────────
depth_bpy = bpy.data.images.load(str(Path(args.depth).resolve()))
depth_bpy.scale(res, res)
px = np.array(depth_bpy.pixels[:], dtype=np.float32)
depth = px[0::4].reshape(res, res)
depth = np.flipud(depth)          # Blender origin is bottom-left
if args.invert_depth:
    depth = 1.0 - depth
depth *= args.depth_scale

h, w = depth.shape

# ── Front face vertices + UVs ─────────────────────────────────────────────────
xs = np.linspace(-1.0,  1.0, w)
ys = np.linspace( 1.0, -1.0, h)
xx, yy = np.meshgrid(xs, ys)

front_verts = np.column_stack([xx.ravel(), yy.ravel(), depth.ravel()])

us = np.linspace(0.0, 1.0, w)
vs = np.linspace(1.0, 0.0, h)
uu, vv = np.meshgrid(us, vs)
front_uvs = np.column_stack([uu.ravel(), vv.ravel()])

# Quad faces for front
i_idx = np.arange(h - 1)
j_idx = np.arange(w - 1)
ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")
base = (ii * w + jj).ravel()
front_faces = np.column_stack([base, base + 1, base + w + 1, base + w])

print(f"[generate_3d] front: {len(front_verts):,} verts  {len(front_faces):,} faces")

# ── Build solid mesh with bmesh ───────────────────────────────────────────────
bm = bmesh.new()
uv_lay = bm.loops.layers.uv.new("UVMap")

# Add front vertices
bm_verts = [bm.verts.new(v) for v in front_verts]
bm.verts.ensure_lookup_table()

# Add front faces
for f in front_faces:
    try:
        face = bm.faces.new([bm_verts[i] for i in f])
        for loop, vi in zip(face.loops, f):
            loop[uv_lay].uv = front_uvs[vi]
        face.smooth = True
    except Exception:
        pass

# ── Side walls (4 edges → extruded inward to z = -depth_scale) ───────────────
back_z = -args.depth_scale * 0.5   # back plate Z position

def add_wall_quad(bm, uv_lay, v0, v1, b0, b1, uv_avg):
    """Add a quad wall face between front edge and back edge."""
    try:
        face = bm.faces.new([v0, v1, b1, b0])
        for loop in face.loops:
            loop[uv_lay].uv = uv_avg
        face.smooth = False
    except Exception:
        pass

# Collect border vertices (top, bottom, left, right edges)
# Top edge: row 0
# Bottom edge: row h-1
# Left edge: col 0
# Right edge: col w-1

back_top    = []
back_bottom = []
back_left   = []
back_right  = []

# Top edge wall
top_back = []
for j in range(w):
    fv = front_verts[0 * w + j]
    bv = bm.verts.new((fv[0], fv[1], back_z))
    top_back.append(bv)
for j in range(w - 1):
    add_wall_quad(bm, uv_lay,
                  bm_verts[j], bm_verts[j + 1],
                  top_back[j], top_back[j + 1],
                  ((j / w + (j+1) / w) / 2, 1.0))

# Bottom edge wall
bottom_back = []
for j in range(w):
    fv = front_verts[(h-1) * w + j]
    bv = bm.verts.new((fv[0], fv[1], back_z))
    bottom_back.append(bv)
for j in range(w - 1):
    add_wall_quad(bm, uv_lay,
                  bm_verts[(h-1)*w + j + 1], bm_verts[(h-1)*w + j],
                  bottom_back[j + 1], bottom_back[j],
                  ((j / w + (j+1) / w) / 2, 0.0))

# Left edge wall
left_back = []
for i in range(h):
    fv = front_verts[i * w + 0]
    bv = bm.verts.new((fv[0], fv[1], back_z))
    left_back.append(bv)
for i in range(h - 1):
    add_wall_quad(bm, uv_lay,
                  bm_verts[(i+1)*w], bm_verts[i*w],
                  left_back[i + 1], left_back[i],
                  (0.0, 1.0 - (i / h)))

# Right edge wall
right_back = []
for i in range(h):
    fv = front_verts[i * w + (w-1)]
    bv = bm.verts.new((fv[0], fv[1], back_z))
    right_back.append(bv)
for i in range(h - 1):
    add_wall_quad(bm, uv_lay,
                  bm_verts[i*w + (w-1)], bm_verts[(i+1)*w + (w-1)],
                  right_back[i], right_back[i + 1],
                  (1.0, 1.0 - (i / h)))

# ── Back plate ────────────────────────────────────────────────────────────────
corners = [
    top_back[0], top_back[-1],
    bottom_back[-1], bottom_back[0],
]
try:
    face = bm.faces.new([corners[0], corners[1], corners[2], corners[3]])
    uv_corners = [(0,1),(1,1),(1,0),(0,0)]
    for loop, uv in zip(face.loops, uv_corners):
        loop[uv_lay].uv = uv
    face.smooth = False
except Exception:
    pass

# ── Finalise mesh ─────────────────────────────────────────────────────────────
mesh = bpy.data.meshes.new("DepthMesh")
bm.to_mesh(mesh)
bm.free()
mesh.update()

obj = bpy.data.objects.new("DepthMesh", mesh)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj
obj.select_set(True)

# ── Subdivision smooth on front face ─────────────────────────────────────────
if args.smooth_levels > 0:
    mod = obj.modifiers.new("Subdiv", "SUBSURF")
    mod.subdivision_type = "SIMPLE"
    mod.levels = args.smooth_levels
    mod.render_levels = args.smooth_levels
    bpy.ops.object.modifier_apply(modifier="Subdiv")

# ── Material ──────────────────────────────────────────────────────────────────
mat = bpy.data.materials.new("DepthMaterial")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
for n in list(nodes):
    nodes.remove(n)

out_node     = nodes.new("ShaderNodeOutputMaterial")
principled   = nodes.new("ShaderNodeBsdfPrincipled")
tex_img_node = nodes.new("ShaderNodeTexImage")
uv_node      = nodes.new("ShaderNodeTexCoord")

tex_image = bpy.data.images.load(str(Path(args.texture).resolve()))
tex_img_node.image = tex_image

principled.inputs["Roughness"].default_value = 0.7
principled.inputs["Metallic"].default_value  = 0.0

links.new(uv_node.outputs["UV"],         tex_img_node.inputs["Vector"])
links.new(tex_img_node.outputs["Color"], principled.inputs["Base Color"])
links.new(principled.outputs["BSDF"],    out_node.inputs["Surface"])

out_node.location     = (400, 0)
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

print(f"\n[generate_3d] Exported: {out}\n")
