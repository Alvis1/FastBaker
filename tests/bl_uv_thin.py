# FastBake UV stage on thin folded parts — regression test.
#
# A 5k-quad retopo of a 700k-triangle car wraps its ~1 cm thin parts (fender
# lips, mirror housings) in two sheets a few millimetres apart, joined by
# knife-edge folds. The UV stage used to
#   * never cut those folds (a convex edge is the most expensive seam), so
#     SLIM laid one sheet over the other: 45-470 overlapping faces;
#   * hand anything above 24 overlapping faces straight to Smart UV Project,
#     which overlapped as well (62-190 faces), and then keep the fallback even
#     where the solver's UVs had fewer bad faces.
#
# The stand-in here is procedural: a hat whose brim is a ruffled wedge, 1 cm
# thick at the root and a knife edge at the rim, meshed with ~6 cm quads like
# the car's low-poly. The old stage reproduced both failures on it (solver 47
# overlapping faces, Smart UV Project 161 — and the 161 were kept).
#
# Groups:
#   1. thin brim: solver UVs kept with zero overlapping / flipped faces
#      (counted independently), and every knife-edge fold is a seam;
#   2. the same brim with fold seams off: the per-cluster overlap repair has
#      work to do and still ends clean on its own;
#   3. both QA gates fail (fold seams and repair off): the set with fewer bad
#      faces is kept — the solver's UVs AND seams are restored and the notice
#      names them; with the solver's count inflated, the fallback is kept;
#   4. a smooth closed part beside a busy one gets a cut of its own (the
#      curvature ranking used to leave it seamless; on the car such a part
#      came out as 527 one-face islands).
#
# Run:
#   blender --background --factory-startup --python-exit-code 1 \
#       --python tests/bl_uv_thin.py

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
FOLD_DEG = 150.0           # knife edges the stage must cut (FOLD_SEAM_ANGLE)

_results = []


def group(name, fn):
    try:
        fn()
    except Exception:
        traceback.print_exc()
        print("FAIL: " + name)
        _results.append((name, False))
    else:
        print("PASS: " + name)
        _results.append((name, True))


# --- meshes -------------------------------------------------------------------

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def _steps(p0, p1, step):
    n = max(1, int(round(math.dist(p0, p1) / step)))
    return [(p0[0] + (p1[0] - p0[0]) * i / n, p0[1] + (p1[1] - p0[1]) * i / n)
            for i in range(n)]


def thin_brim_hat(width=0.3, thick=0.01, ruffle=0.15, waves=7, segments=48,
                  step=0.0625):
    """Closed hat: r = 0.5 m cylinder, 1 m tall, with a brim at z = 0.45 m
    that is a wedge `thick` at the root and a knife edge at the rim, ruffled
    by `ruffle` * sin(waves * angle) (more toward the rim)."""
    r0, zb = 0.5, 0.45
    prof = []
    prof += _steps((0.0, 0.0), (r0, 0.0), step)               # bottom cap
    prof += _steps((r0, 0.0), (r0, zb), step)                 # lower wall
    nb = max(2, int(round(width / step)))
    prof += [(r0 + width * i / nb, zb + thick * 0.5 * i / nb)
             for i in range(nb)]                              # brim underside
    prof += [(r0 + width, zb + thick * 0.5)]                  # knife edge
    prof += [(r0 + width * (nb - i) / nb, zb + thick * (0.5 + 0.5 * i / nb))
             for i in range(1, nb)]                           # brim top
    prof += _steps((r0, zb + thick), (r0, 1.0), step)         # upper wall
    prof += _steps((r0, 1.0), (0.0, 1.0), step) + [(0.0, 1.0)]  # top cap

    # Vertex order matters to the outcome (terminal ties, BFS order), so it is
    # fixed: rings first, then the two axis vertices.
    bm = bmesh.new()
    rings = []
    for k in range(segments):
        th = 2.0 * math.pi * k / segments
        ring = []
        for j, (r, z) in enumerate(prof):
            if r <= 1e-9:
                ring.append(None)
                continue
            lift = ruffle * math.sin(waves * th) * max(0.0, (r - r0) / width)
            ring.append(bm.verts.new((r * math.cos(th), r * math.sin(th),
                                      z + lift)))
        rings.append(ring)
    axis = {j: bm.verts.new((0.0, 0.0, z))
            for j, (r, z) in enumerate(prof) if r <= 1e-9}
    for k in range(segments):
        a, b = rings[k], rings[(k + 1) % segments]
        for j in range(len(prof) - 1):
            quad = []
            for v in (a[j] or axis[j], b[j] or axis[j],
                      b[j + 1] or axis[j + 1], a[j + 1] or axis[j + 1]):
                if v not in quad:
                    quad.append(v)
            if len(quad) >= 3:
                bm.faces.new(quad)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("ThinBrim")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("ThinBrim", me)
    bpy.context.collection.objects.link(obj)
    return obj


def rock_and_ball():
    """A noisy icosphere (curvature extremes everywhere) and, 3 m away, a
    smooth UV sphere, joined into one mesh with two closed parts."""
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.0)
    rock = bpy.context.active_object
    off = Vector((5.1, 1.9, 0.3))
    for v in rock.data.vertices:
        p = v.co * 2.4 + off
        v.co = v.co + v.normal * (noise.noise(p) * 0.32
                                  + noise.noise(p * 2.3) * 0.16)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16,
                                         radius=0.6, location=(3.0, 0.0, 0.0))
    ball = bpy.context.active_object
    bpy.ops.object.select_all(action="DESELECT")
    rock.select_set(True)
    ball.select_set(True)
    bpy.context.view_layer.objects.active = rock
    bpy.ops.object.join()
    obj = bpy.context.active_object
    obj.name = "RockAndBall"
    return obj


# --- independent measurements (not the stage's own helpers) --------------------

def isolate(obj):
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def overlapping_faces(obj):
    from fastbake import compat
    isolate(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        with compat.uv_select_sync(bpy.context.scene.tool_settings):
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.select_overlap()
            bm = bmesh.from_edit_mesh(obj.data)
            return sum(1 for f in bm.faces if f.select)
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")


def uv_state(me):
    """(per-loop UVs in face order, per-edge seam flags) via bmesh."""
    bm = bmesh.new()
    bm.from_mesh(me)
    layer = bm.loops.layers.uv.active
    uvs = np.array([tuple(l[layer].uv) for f in bm.faces for l in f.loops])
    seams = np.array([e.seam for e in bm.edges])
    bm.free()
    return uvs, seams


def flipped_faces(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    layer = bm.loops.layers.uv.active
    areas = []
    for f in bm.faces:
        p = [l[layer].uv for l in f.loops]
        areas.append(0.5 * sum(p[i - 1].x * p[i].y - p[i].x * p[i - 1].y
                               for i in range(len(p))))
    bm.free()
    a = np.array(areas)
    real = np.abs(a) > 1e-9
    return int(min(((a > 0) & real).sum(), ((a < 0) & real).sum()))


def fold_edges(me, deg):
    bm = bmesh.new()
    bm.from_mesh(me)
    lim = math.radians(deg)
    out = [e.index for e in bm.edges
           if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > lim]
    bm.free()
    return out


def run_stage(obj):
    from fastbake import props
    from fastbake.pipeline import uvs
    from fastbake.pipeline.context import PipelineContext
    ctx = PipelineContext(props.settings_to_dict(bpy.context.scene.fastbake),
                          [obj.name], obj.name)
    ctx.base = "Thin"
    ctx.result_name = obj.name
    isolate(obj)
    uvs.UVStage().start(ctx)
    return ctx


def check_clean(obj, ctx):
    stats = ctx.flags["uv_stats"]
    assert not stats["fallback"] and stats.get("kept", "solver") == "solver", \
        "the solver UVs were not kept: %r" % (stats,)
    assert stats["qa"] == "clean", "QA verdict %r" % stats["qa"]
    uvs_, _ = uv_state(obj.data)
    assert np.isfinite(uvs_).all(), "non-finite UVs"
    assert uvs_.min() >= -1e-4 and uvs_.max() <= 1.0 + 1e-4, "UVs outside [0,1]"
    n_ov = overlapping_faces(obj)
    n_fl = flipped_faces(obj.data)
    assert n_ov == 0 and n_fl == 0, \
        "%d overlapping, %d flipped face(s)" % (n_ov, n_fl)


class patched:
    """Temporarily replace module/class attributes."""

    def __init__(self, *triples):
        self.triples = triples

    def __enter__(self):
        self.saved = [(o, n, getattr(o, n)) for o, n, _ in self.triples]
        for o, n, v in self.triples:
            setattr(o, n, v)

    def __exit__(self, *exc):
        for o, n, v in self.saved:
            setattr(o, n, v)


# --- groups -------------------------------------------------------------------

def g1_thin_brim():
    clear_scene()
    obj = thin_brim_hat()
    folds = fold_edges(obj.data, FOLD_DEG)
    assert len(folds) >= 24, "test mesh lost its knife edges (%d)" % len(folds)
    ctx = run_stage(obj)
    check_clean(obj, ctx)
    _, seams = uv_state(obj.data)
    uncut = [ei for ei in folds if not seams[ei]]
    assert not uncut, "%d knife-edge fold(s) left uncut" % len(uncut)
    stats = ctx.flags["uv_stats"]
    assert stats["fold_seams"] == len(folds), \
        "stage cut %d fold edges, mesh has %d" % (stats["fold_seams"], len(folds))
    print("[uv_thin] brim: %d faces, %d fold seams, %d islands, stretch %.3f"
          % (len(obj.data.polygons), stats["fold_seams"], stats["islands"],
             stats["stretch"]))


def g2_repair_alone():
    from fastbake.pipeline import uvs
    clear_scene()
    obj = thin_brim_hat()
    with patched((uvs, "FOLD_SEAM_ANGLE", math.pi)):
        ctx = run_stage(obj)
    stats = ctx.flags["uv_stats"]
    assert stats["fold_seams"] == 0, "fold seams were not switched off"
    assert stats["overlap_fix"] >= 1, \
        "without fold seams the first unwrap no longer overlaps — the brim " \
        "stopped exercising the repair"
    check_clean(obj, ctx)
    print("[uv_thin] brim without fold seams: %d repair round(s), clean"
          % stats["overlap_fix"])


def _both_fail_run(inflate_solver=0):
    """Fold seams and the overlap repair off (the old seam placement), with
    spies on the fallback (solver UVs + seams just before it ran) and on
    _keep_better (both verdicts)."""
    from fastbake.pipeline import uvs
    clear_scene()
    obj = thin_brim_hat()
    seen = {}
    real_fallback = uvs.UVStage._fallback
    real_keep = uvs.UVStage._keep_better
    real_qa = uvs.UVStage._qa

    def spy_fallback(self, ctx, me, ov, tex_size):
        bm = bmesh.from_edit_mesh(me)
        layer = bm.loops.layers.uv.active
        seen["solver_uvs"] = np.array([tuple(l[layer].uv)
                                       for f in bm.faces for l in f.loops])
        seen["solver_seams"] = np.array([e.seam for e in bm.edges])
        return real_fallback(self, ctx, me, ov, tex_size)

    def spy_keep(self, ctx, me, ov, stats, failed):
        seen["fallback_bad"] = stats["bad_faces"]
        seen["solver_bad"] = failed[1]["bad_faces"] if failed and failed[1] else None
        return real_keep(self, ctx, me, ov, stats, failed)

    calls = {"n": 0}

    def qa_inflated(self, ctx, me, ov, stats):
        ok = real_qa(self, ctx, me, ov, stats)
        calls["n"] += 1
        if calls["n"] == 1 and inflate_solver and stats["bad_faces"] is not None:
            stats["flipped"] += inflate_solver
            stats["bad_faces"] += inflate_solver
            stats["qa"] = "%d flipped, %d overlapping face(s)" % (
                stats["flipped"], stats["overlaps"])
            return False
        return ok

    with patched((uvs, "FOLD_SEAM_ANGLE", math.pi),
                 (uvs.UVStage, "_overlap_rounds",
                  lambda self, ctx, me, ov, solver: 0),
                 (uvs.UVStage, "_fallback", spy_fallback),
                 (uvs.UVStage, "_keep_better", spy_keep),
                 (uvs.UVStage, "_qa", qa_inflated)):
        ctx = run_stage(obj)
    return obj, ctx, seen


def g3_keep_solver():
    obj, ctx, seen = _both_fail_run()
    assert "solver_uvs" in seen, "the solver UVs passed QA — no fallback ran"
    assert seen.get("fallback_bad") is not None, \
        "the Smart UV Project fallback passed QA — nothing to choose between"
    assert seen["solver_bad"] < seen["fallback_bad"], \
        "precondition: solver %r vs fallback %r bad faces" % (
            seen["solver_bad"], seen["fallback_bad"])
    stats = ctx.flags["uv_stats"]
    assert stats["kept"] == "solver" and not stats["fallback"], \
        "kept %r (fallback=%r)" % (stats["kept"], stats["fallback"])
    uvs_, seams = uv_state(obj.data)
    assert uvs_.shape == seen["solver_uvs"].shape and \
        np.allclose(uvs_, seen["solver_uvs"], atol=1e-6), \
        "the kept UVs are not the solver's"
    assert seams.any() and (seams == seen["solver_seams"]).all(), \
        "the solver's seams were not restored"
    assert any("kept the solver UVs" in n for n in ctx.notices), ctx.notices
    n_bad = overlapping_faces(obj) + flipped_faces(obj.data)
    assert 0 < n_bad < seen["fallback_bad"], \
        "measured %d bad faces, fallback had %d" % (n_bad, seen["fallback_bad"])
    print("[uv_thin] both gates failed: solver %d vs fallback %d bad faces — "
          "solver kept (measured %d)" % (seen["solver_bad"],
                                          seen["fallback_bad"], n_bad))


def g3_keep_fallback():
    obj, ctx, seen = _both_fail_run(inflate_solver=100000)
    assert seen.get("fallback_bad") is not None and \
        seen["solver_bad"] > seen["fallback_bad"], seen
    stats = ctx.flags["uv_stats"]
    assert stats["kept"] == "fallback" and stats["fallback"], \
        "kept %r (fallback=%r)" % (stats["kept"], stats["fallback"])
    _, seams = uv_state(obj.data)
    assert not seams.any(), "solver seams survived next to the fallback UVs"
    assert any("kept the Smart UV Project fallback" in n for n in ctx.notices), \
        ctx.notices
    print("[uv_thin] solver inflated to %d bad faces: fallback (%d) kept"
          % (seen["solver_bad"], seen["fallback_bad"]))


def g4_closed_part():
    clear_scene()
    obj = rock_and_ball()
    ctx = run_stage(obj)
    check_clean(obj, ctx)
    assert any("closed part" in l for l in ctx.report_lines), \
        "the curvature ranking now reaches the ball by itself — the case " \
        "no longer exercises the closed-part terminals"
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    ball_seams = sum(1 for e in bm.edges
                     if e.seam and min(v.co.x for v in e.verts) > 2.0)
    bm.free()
    assert ball_seams > 0, "the smooth closed part got no seam"
    print("[uv_thin] rock + ball: ball cut by %d seam edge(s)" % ball_seams)


def main():
    import fastbake
    fastbake.register()
    fb = bpy.context.scene.fastbake
    fb.tex_size = TEX_SIZE
    fb.advanced.protect_view = False

    group("1 thin folded brim: clean solver UVs, folds cut", g1_thin_brim)
    group("2 thin folded brim without fold seams: repair recovers",
          g2_repair_alone)
    group("3a both gates fail: solver set kept (UVs + seams)", g3_keep_solver)
    group("3b both gates fail: fallback kept when it has fewer bad faces",
          g3_keep_fallback)
    group("4 smooth closed part next to a busy one gets a cut", g4_closed_part)

    print("\n== uv thin-part summary ==")
    for name, ok in _results:
        print(("PASS  " if ok else "FAIL  ") + name)
    bad = [n for n, ok in _results if not ok]
    if bad:
        print("%d/%d groups FAILED" % (len(bad), len(_results)))
        sys.exit(1)
    print("all %d groups passed" % len(_results))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
