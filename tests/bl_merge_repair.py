# Headless test for the merge stage's small-defect repair and the smooth
# low-poly (fastbake/pipeline/merge.py MergeStage._repair_openings,
# retopo._shade_smooth).
#
# Regression it guards: one boundary edge anywhere used to make the whole
# copy an "open sheet". The entire model was solidified, which inflated the
# scaffold (and so the low-poly) by half the shell thickness and doubled the
# area the voxel face budget paid for. On an imported car a 2 cm flap made
# the low-poly 6 cm too fat everywhere and coarsened the voxel 3.5 -> 4.9 cm.
#
# Every case runs the real driver through retopo (QuadriFlow) and checks the
# merge path, the log, how far the source surface sits from the scaffold, and
# that the result is smooth-shaded.
#
# Run:
#   blender --background --factory-startup --python-exit-code 1 \
#       --python tests/bl_merge_repair.py

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
from mathutils.bvhtree import BVHTree

_results = []
OUTDIR = tempfile.mkdtemp(prefix="fastbake_repair_")


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


# --- scene --------------------------------------------------------------------

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for x in list(coll):
            if x.users == 0:
                coll.remove(x)


def edit(obj, fn):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    fn(bm)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def dense_sphere(name):
    """r = 1 sphere with ~3.3 cm edges: about three merge voxels per edge."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=192, ring_count=96, radius=1.0)
    o = bpy.context.active_object
    o.name = name
    return o


def run(objs, quads=600):
    """Drive preflight -> transforms -> merge -> retopo. Returns
    (ctx, merge stage, world-space scaffold mesh copy, result object)."""
    from fastbake.pipeline import driver, preflight
    mat = bpy.data.materials.get("RepairMat") or bpy.data.materials.new("RepairMat")
    for o in objs:
        if not o.data.materials:
            o.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    fb = bpy.context.scene.fastbake
    fb.quads = quads
    fb.review_stages = False
    fb.advanced.engine = "QUADRIFLOW"
    fb.advanced.output_dir = OUTDIR
    ctx, err = preflight.make_context(bpy.context)
    assert not err, err
    stages = driver.build_bake_stages()[:4]
    stage = next(s for s in stages if s.stage_id == "merge")
    inner = stage.start
    seen = {}

    def spy(c):
        inner(c)
        work = c.obj(c.work_name)
        me = work.data.copy()
        me.transform(work.matrix_world)
        seen["scaffold"] = me
    stage.start = spy
    driver.launch(bpy.context, stages, ctx)
    return ctx, stage, seen["scaffold"], bpy.data.objects[ctx.result_name]


def merge_log(ctx):
    return [line for line in ctx.report_lines if line.startswith("merge:")]


def signed_offsets(objs, scaffold_me):
    """Signed distance of every source vertex to the scaffold: negative =
    the vertex lies inside the scaffold, i.e. the scaffold is inflated."""
    bm = bmesh.new()
    bm.from_mesh(scaffold_me)
    bm.normal_update()
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    out = []
    for o in objs:
        for v in o.data.vertices:
            p = o.matrix_world @ v.co
            loc, nor, _i, d = tree.find_nearest(p)
            out.append(d if (p - loc).dot(nor) > 0.0 else -d)
    return np.array(out)


def voxel_of(ctx):
    for line in merge_log(ctx):
        if line.startswith("merge: voxel ") and "(" in line:
            return float(line.split()[2])
    raise AssertionError("no voxel line in the merge log")


def assert_smooth(result):
    n = len(result.data.polygons)
    smooth = sum(1 for p in result.data.polygons if p.use_smooth)
    assert n > 0 and smooth == n, "result is not smooth-shaded: %d/%d" % (smooth, n)


def assert_unthickened_closed(ctx, stage, objs, scaffold):
    log = merge_log(ctx)
    assert not any("solidified" in line for line in log), \
        "a repaired closed model must not be solidified: %s" % log
    assert stage._path == "reused as-is", "path %r" % stage._path
    off = signed_offsets(objs, scaffold)
    assert np.abs(np.median(off)) < 1e-4, "scaffold moved: median %.5f" % np.median(off)


# --- cases --------------------------------------------------------------------

def main():
    import fastbake
    fastbake.register()

    def stray_flap():
        clear()
        o = dense_sphere("FlapSphere")

        def add_flap(bm):
            ring = [e for e in bm.edges
                    if abs(e.verts[0].co.z) < 1e-4 and abs(e.verts[1].co.z) < 1e-4]
            e = min(ring, key=lambda e: abs(e.verts[0].co.y))
            a, b = e.verts
            apex = bm.verts.new((a.co + b.co) / 2 * 1.015)    # pokes out 1.5 cm
            bm.faces.new((a, b, apex))
        edit(o, add_flap)
        ctx, stage, scaffold, result = run([o])
        log = merge_log(ctx)
        assert any("removed 1 flap(s)" in line for line in log), log
        assert_unthickened_closed(ctx, stage, [o], scaffold)
        # area counted once: a 4*pi sphere, not the doubled open-sheet budget
        area = float([l for l in log if l.startswith("merge: voxel ")][-1]
                     .split("area ")[1].split(",")[0])
        assert area < 13.0, "area double counted: %.3g" % area
        assert_smooth(result)
    group("closed sphere + 1.5 cm flap -> flap removed, not solidified", stray_flap)

    def pin_hole():
        clear()
        o = dense_sphere("HoleSphere")

        def punch(bm):
            top = max(bm.faces, key=lambda f: f.calc_center_median().z)
            bmesh.ops.delete(bm, geom=[top], context="FACES_ONLY")
        edit(o, punch)
        ctx, stage, scaffold, result = run([o])
        assert any("filled 1 hole(s)" in line for line in merge_log(ctx)), merge_log(ctx)
        assert_unthickened_closed(ctx, stage, [o], scaffold)
        assert_smooth(result)
    group("closed sphere with a pin hole -> filled, not solidified", pin_hole)

    def split_seam():
        clear()
        o = dense_sphere("SeamSphere")

        def split(bm):
            seam = [e for e in bm.edges
                    if all(abs(v.co.y) < 1e-6 and v.co.x > 1e-6 for v in e.verts)]
            bmesh.ops.split_edges(bm, edges=seam)
        edit(o, split)
        ctx, stage, scaffold, result = run([o])
        assert any("welded" in line for line in merge_log(ctx)), merge_log(ctx)
        assert_unthickened_closed(ctx, stage, [o], scaffold)
    group("glTF-style split seam -> welded, not solidified", split_seam)

    def cube_with_loose_sheet():
        clear()
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        cube = bpy.context.active_object
        cube.name = "CubeSheet"
        # subdivided so the probe below is mostly face points, not the
        # corners the voxel grid is allowed to round
        mod = cube.modifiers.new("Sub", "SUBSURF")
        mod.subdivision_type = "SIMPLE"
        mod.levels = 3
        bpy.ops.object.modifier_apply(modifier=mod.name)
        cube_verts = len(cube.data.vertices)

        def add_sheet(bm):
            vs = [bm.verts.new((1.2, y, z)) for (y, z) in
                  ((-0.15, -0.15), (0.15, -0.15), (0.15, 0.15), (-0.15, 0.15))]
            bm.faces.new(vs)
        edit(cube, add_sheet)
        ctx, stage, scaffold, _result = run([cube])
        log = merge_log(ctx)
        assert any("split 1 open face(s)" in line for line in log), log
        solid = [line for line in log if "solidified" in line]
        assert len(solid) == 1 and "_open_fbtmp" in solid[0], solid
        # the cube's own surface must sit on the scaffold, not inside it
        probe = bpy.data.meshes.new("CubeProbe")
        probe.from_pydata([v.co.copy() for v in cube.data.vertices[:cube_verts]], [], [])
        holder = bpy.data.objects.new("CubeProbe", probe)
        off = signed_offsets([holder], scaffold)
        assert np.abs(np.median(off)) < 0.25 * voxel_of(ctx), \
            "closed cube was thickened: median %.5f" % np.median(off)
    group("cube + loose sheet in one object -> only the sheet thickened",
          cube_with_loose_sheet)

    def open_bottom_box():
        clear()
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        box = bpy.context.active_object
        box.name = "OpenBox"
        mod = box.modifiers.new("Sub", "SUBSURF")
        mod.subdivision_type = "SIMPLE"
        mod.levels = 3
        bpy.ops.object.modifier_apply(modifier=mod.name)

        def open_bottom(bm):
            bmesh.ops.delete(bm, geom=[f for f in bm.faces
                                       if f.calc_center_median().z < -0.49],
                             context="FACES")
        edit(box, open_bottom)
        ctx, _stage, scaffold, result = run([box])
        log = merge_log(ctx)
        assert any("solidified" in line and "inward" in line for line in log), log
        off = signed_offsets([box], scaffold)
        # the old even shell put every vertex 1.25 voxels inside the scaffold
        assert np.abs(np.median(off)) < 0.25 * voxel_of(ctx), \
            "outer surface moved: median %.5f (voxel %.5f)" % (np.median(off), voxel_of(ctx))
        assert_smooth(result)
    group("open-bottomed box -> thickened inward, outer surface kept",
          open_bottom_box)

    def leaf_cards():
        clear()
        rnd = random.Random(1)
        verts, faces = [], []
        for _ in range(40):
            cx, cy, cz = rnd.uniform(-1, 1), rnd.uniform(-1, 1), rnd.uniform(-1, 1)
            k = len(verts)
            verts += [(cx, cy, cz), (cx + 0.12, cy, cz),
                      (cx + 0.12, cy + 0.12, cz + 0.03), (cx, cy + 0.12, cz + 0.03)]
            faces.append((k, k + 1, k + 2, k + 3))
        me = bpy.data.meshes.new("Leaves")
        me.from_pydata(verts, [], faces)
        me.update()
        o = bpy.data.objects.new("Leaves", me)
        bpy.context.collection.objects.link(o)
        ctx, stage, _scaffold, _result = run([o])
        log = merge_log(ctx)
        # a card's whole outline is short, but it is not a hole: filling it
        # would leave a zero-thickness sheet the voxel grid drops
        assert not any("filled" in line for line in log), log
        assert any("solidified" in line and "evenly" in line for line in log), log
        assert stage._path == "voxel path", stage._path
    group("leaf cards -> outlines not filled, sheets thickened evenly", leaf_cards)

    passed = sum(1 for _n, ok in _results if ok)
    print("\n%d/%d merge-repair groups passed" % (passed, len(_results)))
    if passed != len(_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
