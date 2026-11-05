# Headless test for the "Outer shell" merge mode (fastbake/pipeline/shell.py):
# a house with rooms and furniture must reduce to its exterior hull, openings
# narrower than the probe get sealed, interiors are discarded, and a hull that
# would mostly be invented surface cancels the run with a clear message.
#
# Run:
#   /Applications/Blender.app/Contents/MacOS/Blender --background \
#       --factory-startup --python tests/bl_outer_shell.py

import os
import sys
import tempfile
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bpy
import numpy as np
from mathutils import Vector

_results = []
W, D, H, WALL = 10.0, 8.0, 3.0, 0.2


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


# --- scene ------------------------------------------------------------------------

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.node_groups, bpy.data.images, bpy.data.materials):
        for x in list(coll):
            if x.users == 0:
                coll.remove(x)


def box(name, size, loc, mat=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if mat is not None:
        o.data.materials.append(mat)
    return o


def build_house(door=False, cutaway=False):
    """10 x 8 x 3 m house: slabs, outer walls, two partitions, six pieces of
    furniture. door=True leaves a 0.9 m gap in the front wall; cutaway=True
    removes the whole front wall (a dollhouse view)."""
    clear()
    mat = bpy.data.materials.new("HouseMat")
    mat.use_nodes = True
    objs = [box("Floor", (W, D, WALL), (0, 0, WALL / 2), mat),
            box("Roof", (W, D, WALL), (0, 0, H - WALL / 2), mat),
            box("WallBack", (W, WALL, H), (0, D / 2 - WALL / 2, H / 2), mat),
            box("WallL", (WALL, D, H), (-W / 2 + WALL / 2, 0, H / 2), mat),
            box("WallR", (WALL, D, H), (W / 2 - WALL / 2, 0, H / 2), mat)]
    if cutaway:
        pass
    elif door:
        gap, y = 0.9, -D / 2 + WALL / 2
        left = (W - gap) / 2
        objs += [box("WallFrontL", (left, WALL, H), (-W / 2 + left / 2, y, H / 2), mat),
                 box("WallFrontR", (left, WALL, H), (W / 2 - left / 2, y, H / 2), mat),
                 box("Lintel", (gap, WALL, H - 2.1), (0, y, H - (H - 2.1) / 2), mat)]
    else:
        objs.append(box("WallFront", (W, WALL, H), (0, -D / 2 + WALL / 2, H / 2), mat))
    objs += [box("PartA", (WALL, D - 2 * WALL, H), (-1.5, 0, H / 2), mat),
             box("PartB", (W / 2 - 1.5, WALL, H), (W / 4 + 0.75 - 0.75, 1.0, H / 2), mat)]
    for i, (s, l) in enumerate((((1.8, 0.9, 0.5), (-3.5, 2.0, 0.45)),
                                ((0.6, 0.6, 1.9), (-4.5, -3.2, 1.15)),
                                ((2.0, 1.0, 0.75), (2.5, -2.0, 0.6)),
                                ((0.9, 0.9, 0.9), (3.8, 2.8, 0.65)),
                                ((1.2, 0.5, 2.0), (0.5, 3.5, 1.2)),
                                ((0.4, 0.4, 0.4), (1.0, -3.0, 0.4)))):
        objs.append(box("Furn%d" % i, s, l, mat))
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    return objs


def interior_faces(obj):
    """Faces whose centre lies strictly inside the house envelope: interior
    surfaces that an outer shell must not contain."""
    n = 0
    for p in obj.data.polygons:
        c = obj.matrix_world @ p.center
        if (abs(c.x) < W / 2 - WALL - 0.05 and abs(c.y) < D / 2 - WALL - 0.05
                and WALL + 0.05 < c.z < H - WALL - 0.05):
            n += 1
    return n


def is_watertight(me):
    from fastbake.pipeline import merge
    d = merge._diagnose(me)
    return d["boundary"] == 0 and d["bad"] == 0


# --- module-level tests ----------------------------------------------------------------

class FakeCtx:
    def __init__(self, names):
        self.src_names = names
        self.flags, self.notices, self.report_lines = {}, [], []

    def obj(self, n):
        return bpy.data.objects.get(n)

    def log(self, m):
        self.report_lines.append(str(m))

    def notice(self, m):
        self.notices.append(str(m))


def scaffold(objs):
    from fastbake.pipeline import shell
    bpy.ops.object.select_all(action="DESELECT")
    copies = []
    for o in objs:
        c = o.copy()
        c.data = o.data.copy()
        bpy.context.collection.objects.link(c)
        c.select_set(True)
        copies.append(c)
    bpy.context.view_layer.objects.active = copies[0]
    bpy.ops.object.join()
    work = bpy.context.active_object
    work.name = "ShellWork"
    voxel = shell.bbox_diagonal(work) / 300.0
    shell.voxel_remesh(work, voxel)
    return work, voxel


def main():
    import fastbake
    fastbake.register()
    from fastbake.pipeline import shell
    from fastbake.pipeline.context import StageError

    def closed_house():
        objs = build_house()
        work, voxel = scaffold(objs)
        ctx = FakeCtx([o.name for o in objs])
        st = shell.extract_outer_shell(ctx, work, voxel, 3.0, 15.0)
        assert interior_faces(work) == 0, "interior surfaces survived"
        assert st["bridged_fraction"] < 0.01, "closed house should need no bridging: %.3f" % st["bridged_fraction"]
        assert is_watertight(work.data), "hull is not watertight"
        assert ctx.notices, "no user notice recorded"
    group("closed house -> exterior hull only", closed_house)

    def door_house_sealed():
        objs = build_house(door=True)
        work, voxel = scaffold(objs)
        ctx = FakeCtx([o.name for o in objs])
        st = shell.extract_outer_shell(ctx, work, voxel, 6.0, 15.0)   # r ≈ 0.79 m > door/2
        assert interior_faces(work) == 0, "door was not sealed: %d interior faces" % interior_faces(work)
        assert 0.0 < st["bridged_fraction"] < 0.05, "door bridge should be a few %%: %.3f" % st["bridged_fraction"]
        assert 0.3 < st["worst_deviation"] < 0.6, "worst deviation should be ~door/2: %.2f" % st["worst_deviation"]
        assert is_watertight(work.data)
    group("door house, probe > door/2 -> sealed, interior gone", door_house_sealed)

    def door_house_small_probe():
        objs = build_house(door=True)
        work, voxel = scaffold(objs)
        ctx = FakeCtx([o.name for o in objs])
        shell.extract_outer_shell(ctx, work, voxel, 2.0, 15.0)        # r ≈ 0.26 m < door/2
        assert interior_faces(work) > 1000, "a probe smaller than the door must leave the interior"
    group("door house, probe < door/2 -> interior kept (documented)", door_house_small_probe)

    def strict_limit_cancels():
        objs = build_house(door=True)
        work, voxel = scaffold(objs)
        ctx = FakeCtx([o.name for o in objs])
        try:
            shell.extract_outer_shell(ctx, work, voxel, 6.0, 0.1)
        except StageError as exc:
            assert "stretched too far" in str(exc)
            return
        raise AssertionError("expected StageError for a 0.1% bridging limit")
    group("bridging limit exceeded -> StageError", strict_limit_cancels)

    def cutaway_cancels():
        objs = build_house(cutaway=True)
        work, voxel = scaffold(objs)
        ctx = FakeCtx([o.name for o in objs])
        try:
            shell.extract_outer_shell(ctx, work, voxel, 30.0, 15.0)
        except StageError as exc:
            assert "invented surface" in str(exc)
            return
        raise AssertionError("a missing front wall must not be silently bridged")
    group("cutaway house -> cancels instead of inventing a facade", cutaway_cancels)

    # --- full pipeline through the merge mode ----------------------------------------
    def pipeline_outer_shell():
        from fastbake.pipeline import driver, preflight
        objs = build_house(door=True)
        fb = bpy.context.scene.fastbake
        fb.quads = 1500
        fb.tex_size = "512"
        fb.advanced.output_dir = tempfile.mkdtemp(prefix="fastbake_shell_")
        fb.advanced.merge_mode = "OUTER_SHELL"
        fb.advanced.shell_probe = 6.0
        fb.advanced.shell_max_bridged = 15.0
        ctx, err = preflight.make_context(bpy.context)
        assert not err, err
        t0 = time.time()
        driver.launch(bpy.context, driver.build_bake_stages(), ctx)
        res = bpy.data.objects[ctx.result_name]
        stats = ctx.flags.get("outer_shell")
        assert stats, "outer_shell stats missing from ctx.flags"
        assert interior_faces(res) == 0, "result contains %d interior faces" % interior_faces(res)
        assert any("Outer shell" in n for n in ctx.notices), ctx.notices
        for key in ("color", "normal"):
            assert os.path.isfile(ctx.files[key])
        print("[outer-shell] pipeline: %d result faces, %.1fs, notices: %s"
              % (len(res.data.polygons), time.time() - t0, ctx.notices))
        fb.advanced.merge_mode = "AUTO"
    group("full pipeline with merge_mode=OUTER_SHELL", pipeline_outer_shell)

    print("\n== bl_outer_shell summary ==")
    for name, ok in _results:
        print(("PASS  " if ok else "FAIL  ") + name)
    bad = [n for n, ok in _results if not ok]
    if bad:
        print("%d/%d checks FAILED" % (len(bad), len(_results)))
        sys.exit(1)
    print("all %d checks passed" % len(_results))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
