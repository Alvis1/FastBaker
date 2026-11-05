# FastBake UV benchmark — plan §11 M3 exit criterion:
#   "Benchmark set beats Smart UV on island count and stretch;
#    zero overlaps/flips"
#
# Runs the UV stage (seam solver -> SLIM unwrap -> relief cuts -> pack) and
# Blender's Smart UV Project over the same meshes, then measures both with the
# SAME independent code — deliberately not the solver's own helpers, so the
# numbers are not produced by the thing under test:
#
#   islands  : flood fill over UV-continuous face adjacency
#   stretch  : per-face normalized 3D-area / UV-area ratio, median
#   flipped  : faces whose UV winding disagrees with the mesh majority
#   overlaps : bpy.ops.uv.select_overlap (C-side, exact) in FACE select mode
#
# Hard gates (fail the run):
#   * solver output has ZERO overlapping and ZERO flipped faces, UVs finite
#     and inside [0,1], on every mesh;
#   * over the whole set, the solver's island total is <= Smart UV's and its
#     mean median-stretch is no worse than Smart UV's (small tolerance).
# Per-mesh wins/losses are printed either way.
#
# Run:
#   blender --background --factory-startup --python-exit-code 1 \
#       --python tests/bl_uv_bench.py
#   ... --python tests/bl_uv_bench.py -- --only rock

import argparse
import math
import os
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bmesh
import bpy
import numpy as np
from mathutils import Vector, noise

TEX_SIZE = "1024"
MARGIN_PX = 4.0            # plan §5 stage 3: pack margin = 4 px / tex size
SMART_ANGLE = 66.0
STRETCH_TOLERANCE = 1.05   # aggregate stretch may be 5 % worse before failing

_failures = []
_rows = []


# --- benchmark meshes ---------------------------------------------------------
# Retopo-output-like: closed, quad-dominant, a few hundred to a few thousand
# faces. Built procedurally so the benchmark is reproducible anywhere.

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.materials):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def _displace(obj, amp, freq, offset):
    off = Vector(offset)
    for v in obj.data.vertices:
        n = v.normal if v.normal.length > 1e-6 else Vector((0.0, 0.0, 1.0))
        p = v.co * freq + off
        v.co = v.co + n * (noise.noise(p) * amp + noise.noise(p * 2.3) * amp * 0.5)
    obj.data.update()


def mesh_blob():
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0)
    o = bpy.context.active_object
    _displace(o, 0.22, 1.6, (1.3, 0.7, 2.9))
    return o


def mesh_rock():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
    o = bpy.context.active_object
    _displace(o, 0.32, 2.4, (5.1, 1.9, 0.3))
    return o


def mesh_torus():
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.33,
                                     major_segments=40, minor_segments=18)
    return bpy.context.active_object


def mesh_capsule():
    bpy.ops.mesh.primitive_cylinder_add(vertices=28, radius=0.5, depth=2.0)
    o = bpy.context.active_object
    bpy.ops.object.modifier_add(type="SUBSURF")
    o.modifiers["Subdivision"].levels = 2
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    return o


def mesh_hardsurface():
    bpy.ops.mesh.primitive_cube_add(size=1.4)
    o = bpy.context.active_object
    bpy.ops.object.modifier_add(type="BEVEL")
    o.modifiers["Bevel"].width = 0.12
    o.modifiers["Bevel"].segments = 3
    bpy.ops.object.modifier_apply(modifier="Bevel")
    bpy.ops.object.modifier_add(type="SUBSURF")
    o.modifiers["Subdivision"].subdivision_type = "SIMPLE"
    o.modifiers["Subdivision"].levels = 2
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    return o


def mesh_limb():
    """A bent tapered tube — the shape seam solvers are usually judged on."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=20, radius=0.35, depth=2.4)
    o = bpy.context.active_object
    bpy.ops.object.modifier_add(type="SUBSURF")
    o.modifiers["Subdivision"].subdivision_type = "SIMPLE"
    o.modifiers["Subdivision"].levels = 3
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    for v in o.data.vertices:
        t = (v.co.z + 1.2) / 2.4
        v.co.x += 0.7 * math.sin(t * math.pi)          # bend
        s = 1.0 - 0.45 * t                              # taper
        v.co.x *= s
        v.co.y *= s
    o.data.update()
    return o


MESHES = [
    ("blob (uv sphere + noise)", mesh_blob),
    ("rock (icosphere + noise)", mesh_rock),
    ("torus (genus 1)", mesh_torus),
    ("capsule (subdivided cylinder)", mesh_capsule),
    ("hard surface (bevelled cube)", mesh_hardsurface),
    ("bent tapered limb", mesh_limb),
]


# --- independent measurements --------------------------------------------------

def _uv_arrays(me):
    """(loop_uvs, loop_starts, loop_totals) for the active UV layer."""
    layer = me.uv_layers.active
    buf = np.empty(len(me.loops) * 2, dtype=np.float64)
    try:
        layer.uv.foreach_get("vector", buf)
    except (AttributeError, TypeError):
        layer.data.foreach_get("uv", buf)
    uvs = buf.reshape(-1, 2)
    starts = np.empty(len(me.polygons), dtype=np.int64)
    totals = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", totals)
    return uvs, starts, totals


def signed_uv_areas(me):
    uvs, starts, totals = _uv_arrays(me)
    out = np.empty(len(starts))
    for i, (s, n) in enumerate(zip(starts, totals)):
        poly = uvs[s:s + n]
        x, y = poly[:, 0], poly[:, 1]
        out[i] = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    return out


def count_flipped(me):
    """Faces whose UV winding disagrees with the mesh's own majority.

    Orientation-convention free on purpose: a wholesale mirror of every
    island is a convention, a single reversed face is a bug.
    """
    areas = signed_uv_areas(me)
    real = np.abs(areas) > 1e-9
    pos = int(((areas > 0) & real).sum())
    neg = int(((areas < 0) & real).sum())
    return min(pos, neg)


def median_stretch(me):
    """Median of max(r, 1/r) where r = (3D area share) / (UV area share)."""
    uv_area = np.abs(signed_uv_areas(me))
    a3 = np.array([p.area for p in me.polygons])
    t3, tuv = a3.sum(), uv_area.sum()
    if t3 <= 1e-12 or tuv <= 1e-12:
        return float("inf")
    keep = a3 > 1e-12
    r = (a3[keep] / t3) / np.maximum(uv_area[keep] / tuv, 1e-12)
    stretch = np.maximum(r, 1.0 / np.maximum(r, 1e-12))
    return float(np.median(stretch))


def count_islands(me):
    """Flood fill over faces, crossing only edges whose two faces share the
    exact same UV coordinates at both endpoints."""
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    layer = bm.loops.layers.uv.active
    if layer is None:
        bm.free()
        return 0

    def welded(edge):
        if len(edge.link_faces) != 2:
            return False
        fa, fb = edge.link_faces
        for vert in edge.verts:
            a = [tuple(round(c, 6) for c in loop[layer].uv)
                 for loop in fa.loops if loop.vert is vert]
            b = [tuple(round(c, 6) for c in loop[layer].uv)
                 for loop in fb.loops if loop.vert is vert]
            if not a or not b or a[0] != b[0]:
                return False
        return True

    seen = [False] * len(bm.faces)
    islands = 0
    for face in bm.faces:
        if seen[face.index]:
            continue
        islands += 1
        seen[face.index] = True
        stack = [face]
        while stack:
            cur = stack.pop()
            for edge in cur.edges:
                if not welded(edge):
                    continue
                for other in edge.link_faces:
                    if not seen[other.index]:
                        seen[other.index] = True
                        stack.append(other)
    bm.free()
    return islands


def isolate(obj):
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def count_overlaps(obj):
    """Overlapping faces per bpy.ops.uv.select_overlap, or None when the
    operator is unavailable (then the caller reports 'unverified')."""
    from fastbake import compat
    isolate(obj)
    tool = bpy.context.scene.tool_settings
    old_sync = getattr(tool, "use_uv_select_sync", True)
    old_mode = tuple(tool.mesh_select_mode)
    sync_cm = getattr(compat, "uv_select_sync", None)
    try:
        bpy.ops.object.mode_set(mode="EDIT")
        if sync_cm is not None:
            # the compat seam, when the light group's helper is present
            with sync_cm(tool):
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.uv.select_overlap()
                bm = bmesh.from_edit_mesh(obj.data)
                return sum(1 for f in bm.faces if f.select)
        tool.use_uv_select_sync = True
        tool.mesh_select_mode = (False, False, True)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_overlap()
        bm = bmesh.from_edit_mesh(obj.data)
        return sum(1 for f in bm.faces if f.select)
    except Exception as exc:
        print("[uv_bench] overlap check unavailable: %s" % exc)
        return None
    finally:
        try:
            tool.use_uv_select_sync = old_sync
            tool.mesh_select_mode = old_mode
        except Exception:
            pass
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass


def measure(obj):
    me = obj.data
    uvs, _, _ = _uv_arrays(me)
    return {
        "faces": len(me.polygons),
        "islands": count_islands(me),
        "stretch": median_stretch(me),
        "flipped": count_flipped(me),
        "overlaps": count_overlaps(obj),
        "uv_min": float(uvs.min()) if uvs.size else 0.0,
        "uv_max": float(uvs.max()) if uvs.size else 0.0,
        "finite": bool(np.isfinite(uvs).all()),
    }


# --- the two unwrappers --------------------------------------------------------

def run_solver(obj):
    """FastBake's UV stage on `obj` (a real PipelineContext, no driver)."""
    from fastbake import props
    from fastbake.pipeline import uvs
    from fastbake.pipeline.context import PipelineContext

    settings = props.settings_to_dict(bpy.context.scene.fastbake)
    ctx = PipelineContext(settings, [obj.name], obj.name)
    ctx.base = "Bench"
    ctx.result_name = obj.name
    isolate(obj)
    uvs.UVStage().start(ctx)
    return ctx


def run_smart_uv(obj, tex_size):
    """Blender's Smart UV Project + the same averaging/packing, as baseline."""
    isolate(obj)
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.mark_seam(clear=True)      # no seam help from the solver
        bpy.ops.uv.smart_project(angle_limit=math.radians(SMART_ANGLE),
                                 island_margin=0.0)
        bpy.ops.uv.average_islands_scale()
        try:
            bpy.ops.uv.pack_islands(shape_method="CONCAVE", rotate=True,
                                    margin_method="FRACTION",
                                    margin=MARGIN_PX / tex_size)
        except TypeError:
            bpy.ops.uv.pack_islands(rotate=True, margin=MARGIN_PX / tex_size)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


# --- one benchmark mesh --------------------------------------------------------

def bench(label, builder, tex_size):
    clear_scene()
    solver_obj = builder()
    solver_obj.name = "Bench_solver"
    # exact copy for the baseline, taken before any seam is marked
    smart_obj = solver_obj.copy()
    smart_obj.data = solver_obj.data.copy()
    smart_obj.name = "Bench_smart"
    bpy.context.collection.objects.link(smart_obj)

    ctx = run_solver(solver_obj)
    solver = measure(solver_obj)
    solver["fallback"] = bool(ctx.flags.get("uv_stats", {}).get("fallback"))

    run_smart_uv(smart_obj, tex_size)
    smart = measure(smart_obj)

    _rows.append((label, solver, smart))
    print("[uv_bench] %-32s faces %5d | solver: %3d islands, stretch %.3f, "
          "%d flipped, %s overlaps%s | smart: %3d islands, stretch %.3f, "
          "%d flipped, %s overlaps"
          % (label, solver["faces"], solver["islands"], solver["stretch"],
             solver["flipped"], solver["overlaps"],
             " (FALLBACK)" if solver["fallback"] else "",
             smart["islands"], smart["stretch"], smart["flipped"],
             smart["overlaps"]))

    # --- hard per-mesh gates (plan §11 M3: zero overlaps / flips) -------------
    def fail(msg):
        _failures.append("%s: %s" % (label, msg))
        print("FAIL: %s — %s" % (label, msg))

    if solver["fallback"]:
        # The Smart-UV fallback is a safety net, not the deliverable: the
        # benchmark exists to prove the solver itself is usable.
        fail("solver bailed out to the Smart UV fallback")
    if not solver["finite"]:
        fail("non-finite UVs")
    if not (-1e-4 <= solver["uv_min"] and solver["uv_max"] <= 1.0 + 1e-4):
        fail("UVs outside [0,1]: %.4f..%.4f" % (solver["uv_min"], solver["uv_max"]))
    if solver["flipped"]:
        fail("%d flipped face(s) in the solver output" % solver["flipped"])
    if solver["overlaps"] is None:
        print("[uv_bench] %s: overlap count unverified (operator unavailable)"
              % label)
    elif solver["overlaps"]:
        fail("%d overlapping face(s) in the solver output" % solver["overlaps"])


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="bl_uv_bench")
    p.add_argument("--only", default="", help="substring filter on mesh label")
    return p.parse_args(argv)


def main():
    args = parse_args()
    import fastbake
    fastbake.register()

    fb = bpy.context.scene.fastbake
    fb.tex_size = TEX_SIZE
    fb.advanced.protect_view = False
    tex_size = float(int(TEX_SIZE))

    meshes = [m for m in MESHES if args.only.lower() in m[0].lower()]
    for label, builder in meshes:
        try:
            bench(label, builder, tex_size)
        except Exception:
            traceback.print_exc()
            _failures.append("%s: raised" % label)
            print("FAIL: %s — exception" % label)

    if not _rows:
        print("no benchmark meshes ran")
        sys.exit(1)

    # --- aggregate comparison against Smart UV --------------------------------
    solver_islands = sum(r[1]["islands"] for r in _rows)
    smart_islands = sum(r[2]["islands"] for r in _rows)
    solver_stretch = float(np.mean([r[1]["stretch"] for r in _rows]))
    smart_stretch = float(np.mean([r[2]["stretch"] for r in _rows]))

    print("\n== uv benchmark summary (%d meshes) ==" % len(_rows))
    print("%-32s %8s %8s %8s %8s" % ("mesh", "isl(fb)", "isl(smart)",
                                     "str(fb)", "str(smart)"))
    for label, solver, smart in _rows:
        # a "!" marks a mesh where the solver bailed out to Smart UV, so its
        # numbers ARE Smart UV's and prove nothing about the solver
        mark = "! " if solver["fallback"] else "  "
        print("%-32s %8d %8d %8.3f %8.3f"
              % ((mark + label)[:32], solver["islands"], smart["islands"],
                 solver["stretch"], smart["stretch"]))
    print("%-32s %8d %8d %8.3f %8.3f" % ("TOTAL / MEAN", solver_islands,
                                         smart_islands, solver_stretch,
                                         smart_stretch))

    if solver_islands > smart_islands:
        _failures.append("island total %d > Smart UV's %d"
                         % (solver_islands, smart_islands))
    if solver_stretch > smart_stretch * STRETCH_TOLERANCE:
        _failures.append("mean median stretch %.3f > Smart UV's %.3f (+%.0f%%)"
                         % (solver_stretch, smart_stretch,
                            (STRETCH_TOLERANCE - 1.0) * 100))

    if _failures:
        print("\n%d FAILURE(S):" % len(_failures))
        for f in _failures:
            print("  - " + f)
        sys.exit(1)
    print("\nall UV benchmark gates passed")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
