# FastBake torture set — plan §11 M1 exit criterion:
#   "20-mesh torture set (open shells, self-intersections, kitbash) all
#    produce validated watertight scaffolds"
#
# tests/bl_smoke.py only ever feeds closed primitives, which take the merge
# boolean fast path. Every case here is built procedurally and aims at the
# robust path instead: open shells, thin sheets, non-manifold junctions,
# coincident/self-intersecting geometry, tiny debris, degenerate faces,
# real-world-tiny scale, high genus, mixed selections.
#
# Two assertions per case:
#   1. the MERGE SCAFFOLD is watertight — merge._diagnose(work.data) reports
#      boundary == bad == flipped == 0 (the actual M1 exit criterion);
#   2. the FULL PIPELINE completes — result mesh with faces, a UV layer whose
#      UVs are finite and inside [0,1], and color+normal textures on disk.
#
# Run:
#   blender --background --factory-startup --python-exit-code 1 \
#       --python tests/bl_torture.py
#   ... --python tests/bl_torture.py -- --only thin --quads 300
#
# Prints PASS/FAIL per case and a summary table; exits 1 on any failure.

import argparse
import math
import os
import random
import sys
import tempfile
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, noise

TARGET_QUADS = 500
TEX_SIZE = "512"

_results = []


# --- scene helpers ------------------------------------------------------------

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.images, bpy.data.materials):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def mat(name, color, rough=0.5, metallic=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m


def finish(objs, active, material=None):
    """Select `objs`, make `active` active, give material-less ones a material."""
    for i, o in enumerate(objs):
        if o.type == "MESH" and not o.data.materials:
            o.data.materials.append(material or mat(
                "TortureMat%d" % i, (0.55, 0.5, 0.45, 1.0)))
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active
    return objs, active


def new_obj(name, verts, edges, faces):
    """Build an object from raw pydata — the only way to author deliberately
    broken topology (degenerate faces, loose wire, duplicated shells)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, edges, faces)
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def bm_edit(obj, fn):
    """Run `fn(bm)` over the object's mesh through bmesh."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    fn(bm)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def displace(obj, amp=0.18, freq=2.5, offset=(0.0, 0.0, 0.0)):
    """Sculpt-like multi-frequency Perlin displacement along the normals."""
    off = Vector(offset)
    for v in obj.data.vertices:
        n = v.normal if v.normal.length > 1e-6 else Vector((0.0, 0.0, 1.0))
        p = v.co * freq + off
        d = noise.noise(p) * amp + noise.noise(p * 2.7) * amp * 0.4
        v.co = v.co + n * d
    obj.data.update()


# --- torture cases ------------------------------------------------------------
# Each builder leaves a fresh scene behind and returns (objects, active).

def case_open_plane():
    """A flat grid: zero volume, all-boundary edges. Must be solidified."""
    clear_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=24, y_subdivisions=24, size=2.0)
    g = bpy.context.active_object
    g.name = "OpenPlane"
    for v in g.data.vertices:
        v.co.z = 0.18 * math.sin(v.co.x * 3.0) * math.cos(v.co.y * 3.0)
    g.data.materials.append(mat("PlaneMat", (0.2, 0.6, 0.9, 1.0)))
    return finish([g], g)


def case_open_hemisphere():
    """Sphere with the bottom cap deleted: one big boundary loop."""
    clear_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0)
    s = bpy.context.active_object
    s.name = "OpenHemi"

    def cut(bm):
        faces = [f for f in bm.faces if f.calc_center_median().z < -0.25]
        bmesh.ops.delete(bm, geom=faces, context="FACES")
    bm_edit(s, cut)
    s.data.materials.append(mat("HemiMat", (0.9, 0.3, 0.2, 1.0), rough=0.8))
    return finish([s], s)


def case_nonmanifold_fin():
    """Closed cube plus a fin face sharing an edge: non-manifold T-junction."""
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.5)
    c = bpy.context.active_object
    c.name = "FinCube"

    def add_fin(bm):
        bm.verts.ensure_lookup_table()
        top = sorted(bm.verts, key=lambda v: -v.co.z)[:2]
        v1 = bm.verts.new((top[0].co.x, top[0].co.y, top[0].co.z + 0.9))
        v2 = bm.verts.new((top[1].co.x, top[1].co.y, top[1].co.z + 0.9))
        bm.faces.new((top[0], top[1], v2, v1))
    bm_edit(c, add_fin)
    c.data.materials.append(mat("FinMat", (0.4, 0.8, 0.3, 1.0)))
    return finish([c], c)


def case_kitbash_debris():
    """Five overlapping primitives + two tiny loose specks (debris cleanup)."""
    clear_scene()
    objs = []
    bpy.ops.mesh.primitive_cube_add(size=1.4, location=(0, 0, 0))
    body = bpy.context.active_object
    body.name = "KitBody"
    body.data.materials.append(mat("KitA", (0.8, 0.7, 0.2, 1.0), metallic=1.0))
    objs.append(body)

    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.35, depth=2.2,
                                        location=(0.3, 0.2, 0.3))
    cyl = bpy.context.active_object
    cyl.name = "KitCyl"
    cyl.rotation_euler = (0.4, 0.9, 0.0)
    objs.append(cyl)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.7,
                                         location=(-0.5, 0.1, 0.5))
    sph = bpy.context.active_object
    sph.name = "KitSphere"
    sph.data.materials.append(mat("KitB", (0.2, 0.3, 0.9, 1.0), rough=0.25))
    objs.append(sph)

    bpy.ops.mesh.primitive_torus_add(major_radius=0.8, minor_radius=0.18,
                                     location=(0.0, 0.0, -0.4))
    torus = bpy.context.active_object
    torus.name = "KitTorus"
    objs.append(torus)

    for i, loc in enumerate(((2.6, 2.4, 1.9), (-2.7, -2.2, -1.6))):
        bpy.ops.mesh.primitive_cube_add(size=0.05, location=loc)
        d = bpy.context.active_object
        d.name = "KitDebris%d" % i
        objs.append(d)

    return finish(objs, body)


def case_thin_sheet():
    """Very thin closed slab (0.016 units thick): thin-shell voxel sizing."""
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    t = bpy.context.active_object
    t.name = "ThinSheet"
    t.scale = (1.6, 1.2, 0.008)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.modifier_add(type="SUBSURF")
    t.modifiers["Subdivision"].subdivision_type = "SIMPLE"
    t.modifiers["Subdivision"].levels = 3
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    t.data.materials.append(mat("ThinMat", (0.6, 0.6, 0.6, 1.0)))
    return finish([t], t)


def case_self_intersecting_torii():
    """Two torii threaded through each other: heavy self-intersection."""
    clear_scene()
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.3,
                                     major_segments=32, minor_segments=14)
    a = bpy.context.active_object
    a.name = "TorusA"
    a.data.materials.append(mat("TorusMatA", (0.85, 0.2, 0.3, 1.0)))
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.3,
                                     major_segments=32, minor_segments=14,
                                     location=(0.9, 0.0, 0.0),
                                     rotation=(math.pi / 2, 0.0, 0.0))
    b = bpy.context.active_object
    b.name = "TorusB"
    b.data.materials.append(mat("TorusMatB", (0.2, 0.4, 0.85, 1.0), rough=0.2))
    return finish([a, b], a)


def case_open_tube():
    """Cylinder with both caps removed: two boundary loops, zero volume."""
    clear_scene()
    bpy.ops.mesh.primitive_cylinder_add(vertices=28, radius=0.6, depth=2.0,
                                        end_fill_type="NOTHING")
    c = bpy.context.active_object
    c.name = "OpenTube"
    c.data.materials.append(mat("TubeMat", (0.3, 0.75, 0.6, 1.0)))
    return finish([c], c)


def case_twisted_ribbon():
    """A twisted open ribbon (single quad strip): thin, open, curved."""
    clear_scene()
    verts, faces = [], []
    steps = 48
    for i in range(steps + 1):
        t = i / steps
        ang = t * math.tau
        twist = t * math.pi
        centre = Vector((math.cos(ang) * 1.2, math.sin(ang) * 1.2, 0.0))
        side = Vector((math.cos(ang) * math.cos(twist),
                       math.sin(ang) * math.cos(twist),
                       math.sin(twist))) * 0.35
        verts.append(tuple(centre + side))
        verts.append(tuple(centre - side))
        if i:
            k = (i - 1) * 2
            faces.append((k, k + 1, k + 3, k + 2))
    r = new_obj("Ribbon", verts, [], faces)
    r.data.materials.append(mat("RibbonMat", (0.9, 0.8, 0.3, 1.0)))
    return finish([r], r)


def case_open_cone():
    """Cone without its base: a boundary loop meeting a pole."""
    clear_scene()
    bpy.ops.mesh.primitive_cone_add(vertices=28, radius1=0.9, depth=1.8,
                                    end_fill_type="NOTHING")
    c = bpy.context.active_object
    c.name = "OpenCone"
    c.data.materials.append(mat("ConeMat", (0.75, 0.45, 0.85, 1.0)))
    return finish([c], c)


def case_coplanar_cubes():
    """Two cubes sharing one face exactly: coplanar boolean degeneracy."""
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    a = bpy.context.active_object
    a.name = "CoplanarA"
    a.data.materials.append(mat("CoplanarMatA", (0.8, 0.35, 0.2, 1.0)))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(1.0, 0.0, 0.0))
    b = bpy.context.active_object
    b.name = "CoplanarB"
    b.data.materials.append(mat("CoplanarMatB", (0.2, 0.55, 0.8, 1.0)))
    return finish([a, b], a)


def case_debris_swarm():
    """One real shell plus twelve scattered specks: loose-part cleanup."""
    clear_scene()
    rnd = random.Random(7)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
    main = bpy.context.active_object
    main.name = "SwarmMain"
    main.data.materials.append(mat("SwarmMat", (0.5, 0.6, 0.7, 1.0)))
    objs = [main]
    for i in range(12):
        loc = (rnd.uniform(-3, 3), rnd.uniform(-3, 3), rnd.uniform(-3, 3))
        bpy.ops.mesh.primitive_cube_add(size=rnd.uniform(0.02, 0.06), location=loc)
        d = bpy.context.active_object
        d.name = "Speck%02d" % i
        objs.append(d)
    return finish(objs, main)


def case_sculpt_blob():
    """Dense noise-displaced icosphere: the canonical sculpt input."""
    clear_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=1.0)
    s = bpy.context.active_object
    s.name = "SculptBlob"
    displace(s, amp=0.25, freq=1.8, offset=(3.1, 1.7, 0.4))
    s.data.materials.append(mat("BlobMat", (0.6, 0.4, 0.3, 1.0), rough=0.85))
    return finish([s], s)


def case_inconsistent_normals():
    """Closed sphere with half of its faces reversed: inconsistent winding."""
    clear_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0)
    s = bpy.context.active_object
    s.name = "FlippedHalf"

    def flip(bm):
        faces = [f for f in bm.faces if f.calc_center_median().x > 0.0]
        bmesh.ops.reverse_faces(bm, faces=faces)
    bm_edit(s, flip)
    s.data.materials.append(mat("FlipMat", (0.35, 0.35, 0.9, 1.0)))
    return finish([s], s)


def case_duplicate_shells():
    """Two perfectly coincident spheres: fully duplicated geometry."""
    clear_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=14, radius=1.0)
    a = bpy.context.active_object
    a.name = "DupA"
    a.data.materials.append(mat("DupMat", (0.7, 0.7, 0.2, 1.0)))
    bpy.ops.object.duplicate()
    b = bpy.context.active_object
    b.name = "DupB"
    return finish([a, b], a)


def case_spiky():
    """Sphere with extruded spikes: thin protrusions the voxel path can eat."""
    clear_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=1.0)
    s = bpy.context.active_object
    s.name = "Spiky"

    def spike(bm):
        rnd = random.Random(11)
        targets = [f for f in bm.faces if rnd.random() < 0.25]
        for f in targets:
            res = bmesh.ops.extrude_discrete_faces(bm, faces=[f])
            for nf in res["faces"]:
                d = nf.normal.copy() * 0.55
                bmesh.ops.translate(bm, verts=nf.verts, vec=d)
                bmesh.ops.scale(bm, verts=nf.verts, vec=(0.25, 0.25, 0.25),
                                space=Matrix.Translation(
                                    -nf.calc_center_median()))
    bm_edit(s, spike)
    s.data.materials.append(mat("SpikeMat", (0.9, 0.5, 0.15, 1.0)))
    return finish([s], s)


def case_nested_shells():
    """A shell inside a shell (hollow object): interior geometry to discard."""
    clear_scene()
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1.0)
    outer = bpy.context.active_object
    outer.name = "NestedOuter"
    outer.data.materials.append(mat("NestedMat", (0.4, 0.7, 0.4, 1.0)))
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=0.55)
    inner = bpy.context.active_object
    inner.name = "NestedInner"
    return finish([outer, inner], outer)


def case_holed_grid():
    """Grid with every other face deleted: dozens of boundary loops."""
    clear_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=16, y_subdivisions=16, size=2.0)
    g = bpy.context.active_object
    g.name = "HoledGrid"

    def punch(bm):
        bm.faces.ensure_lookup_table()
        rnd = random.Random(3)
        doomed = [f for f in bm.faces if rnd.random() < 0.35]
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm_edit(g, punch)
    for v in g.data.vertices:
        v.co.z = 0.12 * math.sin(v.co.x * 4.0 + v.co.y * 2.0)
    g.data.materials.append(mat("HoledMat", (0.3, 0.3, 0.35, 1.0)))
    return finish([g], g)


def case_mixed_open_closed():
    """One open sheet + one closed solid + one open hemisphere, together."""
    clear_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=12, y_subdivisions=12, size=2.4)
    sheet = bpy.context.active_object
    sheet.name = "MixSheet"
    for v in sheet.data.vertices:
        v.co.z = -0.8 + 0.1 * math.cos(v.co.x * 3.0)
    sheet.data.materials.append(mat("MixMatA", (0.25, 0.5, 0.7, 1.0)))

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.4, 0.2, 0.2))
    cube = bpy.context.active_object
    cube.name = "MixCube"
    cube.data.materials.append(mat("MixMatB", (0.8, 0.8, 0.8, 1.0), metallic=1.0))

    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.8,
                                         location=(-0.5, 0.0, 0.4))
    dome = bpy.context.active_object
    dome.name = "MixDome"

    def cut(bm):
        faces = [f for f in bm.faces if f.calc_center_median().z < 0.4]
        bmesh.ops.delete(bm, geom=faces, context="FACES")
    bm_edit(dome, cut)
    return finish([sheet, cube, dome], cube)


def case_rod_cluster():
    """Six long, very thin rods crossing: extreme aspect ratio."""
    clear_scene()
    rnd = random.Random(5)
    objs = []
    for i in range(6):
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.035, depth=3.0)
        rod = bpy.context.active_object
        rod.name = "Rod%d" % i
        rod.rotation_euler = (rnd.uniform(0, math.pi), rnd.uniform(0, math.pi),
                              rnd.uniform(0, math.pi))
        rod.location = (rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2),
                        rnd.uniform(-0.2, 0.2))
        objs.append(rod)
    objs[0].data.materials.append(mat("RodMat", (0.65, 0.65, 0.7, 1.0),
                                      metallic=1.0, rough=0.3))
    return finish(objs, objs[0])


def case_tiny_scale():
    """A 2 cm object at real-world scale: near QuadriFlow's 1e-4 edge epsilon."""
    clear_scene()
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=0.01)
    s = bpy.context.active_object
    s.name = "TinyScale"
    displace(s, amp=0.0022, freq=180.0, offset=(0.9, 2.2, 5.5))
    s.data.materials.append(mat("TinyMat", (0.5, 0.45, 0.4, 1.0)))
    return finish([s], s)


def case_degenerate_faces():
    """Hand-built strip containing zero-area faces and duplicate vertices."""
    clear_scene()
    verts, faces = [], []
    for i in range(12):
        x = i * 0.25
        verts += [(x, 0.0, 0.0), (x, 0.5, 0.0)]
    # a duplicated vertex pair -> two zero-area quads in the middle
    verts += [verts[10], verts[11]]
    for i in range(11):
        k = i * 2
        faces.append((k, k + 1, k + 3, k + 2))
    faces.append((10, 11, 25, 24))   # zero area: coincident with verts 10/11
    faces.append((24, 25, 13, 12))
    g = new_obj("Degenerate", verts, [], faces)
    g.data.materials.append(mat("DegenMat", (0.7, 0.2, 0.5, 1.0)))
    return finish([g], g)


def case_loose_wire():
    """Cube plus loose edges and loose vertices: wire non-manifold junk."""
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    c = bpy.context.active_object
    c.name = "WireCube"

    def junk(bm):
        bm.verts.ensure_lookup_table()
        anchor = bm.verts[0]               # grab it BEFORE new() invalidates
        a = bm.verts.new((1.2, 0.0, 0.0))  # the index table
        b = bm.verts.new((1.8, 0.4, 0.3))
        bm.edges.new((a, b))
        bm.edges.new((anchor, a))          # wire edge hanging off the cube
        bm.verts.new((-1.6, -1.2, 0.8))    # fully loose vertex
    bm_edit(c, junk)
    c.data.materials.append(mat("WireMat", (0.45, 0.65, 0.25, 1.0)))
    return finish([c], c)


def case_high_genus():
    """Three interlocking torii: genus-3 input for the seam solver."""
    clear_scene()
    objs = []
    for i in range(3):
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.9, minor_radius=0.22, major_segments=28,
            minor_segments=12, location=(i * 0.75 - 0.75, 0.0, 0.0),
            rotation=(0.0, (i % 2) * math.pi / 2, 0.0))
        t = bpy.context.active_object
        t.name = "Genus%d" % i
        objs.append(t)
    objs[0].data.materials.append(mat("GenusMat", (0.2, 0.7, 0.75, 1.0)))
    return finish(objs, objs[0])


def case_long_strip():
    """A very long, very narrow open strip: worst case for UV packing too."""
    clear_scene()
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=64, y_subdivisions=3, size=2.0)
    g = bpy.context.active_object
    g.name = "LongStrip"
    g.scale = (3.0, 0.06, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    for v in g.data.vertices:
        v.co.z = 0.25 * math.sin(v.co.x * 1.5)
    g.data.materials.append(mat("StripMat", (0.85, 0.75, 0.55, 1.0)))
    return finish([g], g)


CASES = [
    ("open plane (zero volume)", case_open_plane),
    ("open hemisphere (boundary loop)", case_open_hemisphere),
    ("non-manifold fin on cube", case_nonmanifold_fin),
    ("kitbash + loose debris", case_kitbash_debris),
    ("thin sheet (0.016 thick)", case_thin_sheet),
    ("self-intersecting torii", case_self_intersecting_torii),
    ("open tube (two boundary loops)", case_open_tube),
    ("twisted open ribbon", case_twisted_ribbon),
    ("open cone (no base)", case_open_cone),
    ("coplanar face-sharing cubes", case_coplanar_cubes),
    ("debris swarm (12 specks)", case_debris_swarm),
    ("sculpt blob (noise icosphere)", case_sculpt_blob),
    ("inconsistent normals (half flipped)", case_inconsistent_normals),
    ("duplicate coincident shells", case_duplicate_shells),
    ("spiky extrusions", case_spiky),
    ("nested shells (hollow)", case_nested_shells),
    ("holed grid (many boundaries)", case_holed_grid),
    ("mixed open + closed selection", case_mixed_open_closed),
    ("thin rod cluster", case_rod_cluster),
    ("tiny real-world scale (2 cm)", case_tiny_scale),
    ("degenerate zero-area faces", case_degenerate_faces),
    ("loose wire edges + verts", case_loose_wire),
    ("genus-3 interlocking torii", case_high_genus),
    ("long narrow strip", case_long_strip),
]


# --- assertions ---------------------------------------------------------------

def read_uvs(mesh):
    layer = mesh.uv_layers.active
    buf = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    try:
        layer.uv.foreach_get("vector", buf)
    except (AttributeError, TypeError):
        layer.data.foreach_get("uv", buf)
    return buf.reshape(-1, 2)


def run_case(label, builder, outdir, quads):
    from fastbake.pipeline import driver, merge, preflight

    builder()
    fb = bpy.context.scene.fastbake
    fb.quads = quads
    fb.tex_size = TEX_SIZE
    fb.review_stages = False
    fb.advanced.output_dir = outdir
    fb.advanced.ssaa = "1"          # keep the bake leg cheap; SSAA is pack's job

    ctx, err = preflight.make_context(bpy.context)
    if err:
        raise AssertionError("make_context refused: %s" % err)

    # Watch the merge scaffold: MergeStage is synchronous, so the work object
    # exists the moment start() returns, and the retopo stage deletes it again
    # once it has consumed it. Wrapping the instance keeps the run itself on
    # the real driver path.
    stages = driver.build_bake_stages()
    merge_stage = next(s for s in stages if s.stage_id == "merge")
    inner_start = merge_stage.start
    seen = {}

    def spy_start(c):
        inner_start(c)
        work = c.obj(c.work_name)
        seen["work"] = c.work_name
        seen["diag"] = merge._diagnose(work.data) if work is not None else None
    merge_stage.start = spy_start

    pipeline_error = None
    try:
        driver.launch(bpy.context, stages, ctx)
    except Exception as exc:          # driver raises RuntimeError on failure
        pipeline_error = exc

    # 1. watertight scaffold (M1 exit criterion) -- checked even if a later
    #    stage blew up, because that is the more informative failure.
    diag = seen.get("diag")
    assert diag is not None, "merge stage produced no work object"
    assert diag["faces"] > 0, "merge scaffold %r has no faces" % seen["work"]
    assert (diag["boundary"] == 0 and diag["bad"] == 0 and diag["flipped"] == 0), \
        ("merge scaffold not watertight: %d boundary, %d bad, %d flipped edges "
         "(%d faces)" % (diag["boundary"], diag["bad"], diag["flipped"],
                         diag["faces"]))

    # 2. the full pipeline completes
    if pipeline_error is not None:
        raise AssertionError("pipeline failed after a watertight merge: %s"
                             % pipeline_error)
    result = bpy.data.objects.get(ctx.result_name or "")
    assert result is not None, "no result object"
    npoly = len(result.data.polygons)
    assert npoly > 0, "result mesh has no faces"
    assert len(result.data.uv_layers) >= 1, "result has no UV layer"
    uvs = read_uvs(result.data)
    assert np.isfinite(uvs).all(), "NaN/inf in UVs"
    assert uvs.min() >= -1e-4 and uvs.max() <= 1.0 + 1e-4, \
        "UVs outside [0,1]: %.4f..%.4f" % (uvs.min(), uvs.max())
    for key in ("color", "normal"):
        path = ctx.files.get(key)
        assert path and os.path.isfile(path), "missing %s texture on disk" % key

    quad_n = sum(1 for p in result.data.polygons if len(p.vertices) == 4)
    print("[torture] %-38s scaffold %5d faces (watertight) -> %5d faces "
          "(%d quads, %.0f%%), engine %s"
          % (label, diag["faces"], npoly, quad_n,
             100.0 * quad_n / max(1, npoly), ctx.flags.get("engine_used", "?")))
    if ctx.notices:
        print("[torture]   notices: %s" % "; ".join(ctx.notices))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="bl_torture")
    p.add_argument("--only", default="",
                   help="run only cases whose label contains this substring")
    p.add_argument("--quads", type=int, default=TARGET_QUADS)
    return p.parse_args(argv)


def main():
    args = parse_args()
    import fastbake
    fastbake.register()

    outdir = tempfile.mkdtemp(prefix="fastbake_torture_")
    cases = [c for c in CASES if args.only.lower() in c[0].lower()]
    print("[torture] %d cases, %d quads, output dir: %s"
          % (len(cases), args.quads, outdir))
    assert len(CASES) >= 20, "the M1 torture set must hold at least 20 meshes"

    for label, builder in cases:
        try:
            run_case(label, builder, outdir, args.quads)
        except Exception:
            traceback.print_exc()
            print("FAIL: " + label)
            _results.append((label, False))
        else:
            print("PASS: " + label)
            _results.append((label, True))

    print("\n== torture summary ==")
    for label, ok in _results:
        print(("PASS  " if ok else "FAIL  ") + label)
    bad = [l for l, ok in _results if not ok]
    if bad:
        print("%d/%d cases FAILED" % (len(bad), len(_results)))
        sys.exit(1)
    print("all %d cases passed" % len(_results))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
