# FastBake headless integration smoke test.
#
# Run:
#   /Applications/Blender.app/Contents/MacOS/Blender --background \
#       --factory-startup --python tests/bl_smoke.py
#
# Builds a small scene (two overlapping cubes + a ~2k-face UV sphere), runs
# the full bake pipeline through the driver's blocking path, then the re-bake
# guard + force path, then the lightmap pipeline. Prints PASS/FAIL per
# assertion group and exits 1 on any failure.

import os
import sys
import tempfile
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bpy
import numpy as np

TARGET_QUADS = 1000

_results = []


def group(name, fn):
    """Run one assertion group; record and print PASS/FAIL, keep going."""
    try:
        fn()
    except Exception:
        traceback.print_exc()
        print("FAIL: " + name)
        _results.append((name, False))
    else:
        print("PASS: " + name)
        _results.append((name, True))


# --- scene building -----------------------------------------------------------

def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def make_material(name, base_color, roughness=0.5, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def build_scene():
    """Two overlapping cubes + one UV sphere (~2k faces), sphere active."""
    clear_scene()

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    cube_a = bpy.context.active_object
    cube_a.name = "SmokeCubeA"
    cube_a.data.materials.append(make_material(
        "SmokeMetal", (0.7, 0.7, 0.75, 1.0), roughness=0.35, metallic=1.0))

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.45, 0.3, 0.25))
    cube_b = bpy.context.active_object
    cube_b.name = "SmokeCubeB"
    # non-uniform scale + rotation: exercises the apply-transforms stage
    cube_b.scale = (1.0, 1.6, 0.55)
    cube_b.rotation_euler = (0.35, 0.0, 0.8)

    # 64x32 UV sphere = 2048 faces, overlaps both cubes
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32,
                                         radius=0.8, location=(0.0, 0.0, 0.7))
    sphere = bpy.context.active_object
    sphere.name = "SmokeSphere"
    sphere.data.materials.append(make_material(
        "SmokeOrange", (1.0, 0.45, 0.1, 1.0), roughness=0.7))

    bpy.ops.object.select_all(action="DESELECT")
    for obj in (cube_a, cube_b, sphere):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = sphere
    return cube_a, cube_b, sphere


def configure_settings(outdir):
    fb = bpy.context.scene.fastbake
    fb.quads = TARGET_QUADS
    fb.tex_size = "512"
    fb.advanced.output_dir = outdir
    # keep the lightmap leg fast; defaults would render 1024px at 64 samples
    fb.advanced.lm_size = "256"
    fb.advanced.lm_samples = 8


def reselect(names, active_name):
    """Unhide + select the given objects, make active_name active."""
    bpy.ops.object.select_all(action="DESELECT")
    for name in names:
        obj = bpy.data.objects[name]
        try:
            obj.hide_set(False)
        except RuntimeError:
            pass  # not in the current view layer
        obj.hide_render = False
        obj.hide_viewport = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[active_name]


def read_uvs(mesh):
    layer = mesh.uv_layers.active
    buf = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    try:
        layer.uv.foreach_get("vector", buf)   # 3.5+ attribute API
    except (AttributeError, TypeError):
        layer.data.foreach_get("uv", buf)     # legacy MeshUVLoop path
    return buf.reshape(-1, 2)


# --- suite ----------------------------------------------------------------------

def main():
    import fastbake
    fastbake.register()
    from fastbake import naming
    from fastbake.pipeline import driver, preflight

    cube_a, cube_b, sphere = build_scene()
    src_names = [cube_a.name, cube_b.name, sphere.name]
    outdir = tempfile.mkdtemp(prefix="fastbake_smoke_")
    configure_settings(outdir)
    print("[bl_smoke] output dir: " + outdir)

    # --- first full bake ------------------------------------------------------
    ctx, err = preflight.make_context(bpy.context)
    assert not err, "make_context failed: %s" % err
    assert ctx is not None
    base = ctx.base
    assert base == "SmokeSphere", "unexpected base name: %r" % base
    launched = driver.launch(bpy.context, driver.build_bake_stages(), ctx)
    assert launched, "driver.launch refused to start"

    def check_result_mesh():
        assert ctx.result_name, "ctx.result_name not set"
        result = bpy.data.objects.get(ctx.result_name)
        assert result is not None, "result %s not in bpy.data.objects" % ctx.result_name
        assert result.type == "MESH"
        n = len(result.data.polygons)
        assert TARGET_QUADS / 3 <= n <= TARGET_QUADS * 3, \
            "poly count %d not within 3x of %d" % (n, TARGET_QUADS)
    group("first-run: result mesh + poly count", check_result_mesh)

    def check_uvs():
        mesh = bpy.data.objects[ctx.result_name].data
        assert len(mesh.uv_layers) >= 1, "result has no UV layer"
        uvs = read_uvs(mesh)
        assert uvs.size > 0, "empty UV data"
        assert np.isfinite(uvs).all(), "NaN/inf in UVs"
        assert uvs.min() >= -1e-4 and uvs.max() <= 1.0 + 1e-4, \
            "UVs outside [0,1]: min %.5f max %.5f" % (uvs.min(), uvs.max())
    group("first-run: UVs inside [0,1], no NaNs", check_uvs)

    def check_textures():
        assert ctx.run == 0, "first run counter should be 0, got %r" % ctx.run
        for key in ("color", "normal"):
            img_name = naming.tex_name(base, key, ctx.run)
            assert img_name in bpy.data.images, "image %s missing (have: %s)" % (
                img_name, [i.name for i in bpy.data.images if i.name.startswith("T_")])
            path = ctx.files.get(key)
            assert path, "ctx.files[%r] not recorded" % key
            assert os.path.isfile(path), "texture file missing on disk: %s" % path
            assert os.path.abspath(path).startswith(os.path.abspath(outdir)), \
                "texture saved outside output dir: %s" % path
    group("first-run: texture images + files on disk", check_textures)

    def check_manifest_material():
        result = bpy.data.objects[ctx.result_name]
        manifest = result.get("fastbake")
        assert manifest is not None, "result['fastbake'] manifest missing"
        keys = set(manifest.keys())
        assert "version" in keys and "sources" in keys, \
            "manifest incomplete, keys: %s" % sorted(keys)
        expected_mat = naming.mat_name(base, ctx.run)
        mats = [s.material.name for s in result.material_slots if s.material]
        assert expected_mat in mats, \
            "material %s not assigned (slots: %s)" % (expected_mat, mats)
    group("first-run: manifest + material assigned", check_manifest_material)

    def check_sources_hidden():
        for name in src_names:
            obj = bpy.data.objects.get(name)
            assert obj is not None, "source %s vanished" % name
            assert obj.hide_render, "source %s not render-hidden" % name
            try:
                hidden = obj.hide_get()
            except RuntimeError:
                hidden = True  # moved out of the view layer counts as hidden
            assert hidden, "source %s still visible in viewport" % name
    group("first-run: sources hidden", check_sources_hidden)

    # --- second run: re-bake guard, then force ---------------------------------
    reselect(src_names, "SmokeSphere")
    ctx2, err2 = preflight.make_context(bpy.context)
    assert not err2, "second make_context failed: %s" % err2

    def check_already_baked():
        msg = ctx2.flags.get("already_baked")
        assert msg, "already_baked flag not set for an unchanged re-run"
        assert "texture" in str(msg).lower(), \
            "already_baked message should mention textures: %r" % msg
    group("re-bake: already_baked guard", check_already_baked)

    ctx2.flags["force"] = True
    launched = driver.launch(bpy.context, driver.build_bake_stages(), ctx2)
    assert launched, "forced driver.launch refused to start"

    def check_second_run_names():
        expected = base + "_02"
        assert ctx2.result_name == expected, \
            "expected result %s, got %s" % (expected, ctx2.result_name)
        assert ctx2.result_name in bpy.data.objects
        for key in ("color", "normal"):
            img_name = naming.tex_name(base, key, 1)  # forced re-bake -> run 1
            assert img_name in bpy.data.images, "image %s missing (have: %s)" % (
                img_name, [i.name for i in bpy.data.images if i.name.startswith("T_")])
    group("re-bake: _02 mesh + _01 textures", check_second_run_names)

    # --- lightmap on the latest result ------------------------------------------
    result_name = ctx2.result_name or ctx.result_name
    assert result_name, "no result object available for the lightmap leg"
    reselect([result_name], result_name)

    ctx3, err3 = preflight.make_context(bpy.context)
    assert not err3, "lightmap make_context failed: %s" % err3
    launched = driver.launch(bpy.context, driver.build_lightmap_stages(), ctx3)
    assert launched, "lightmap driver.launch refused to start"

    def check_lightmap():
        lm_name = naming.tex_name(ctx3.base, "lightmap", ctx3.run)
        assert lm_name in bpy.data.images, "lightmap image %s missing (have: %s)" % (
            lm_name, [i.name for i in bpy.data.images if i.name.startswith("T_")])
        path = ctx3.files.get("lightmap")
        assert path, "ctx.files['lightmap'] not recorded"
        assert path.lower().endswith(".exr"), "lightmap is not an EXR: %s" % path
        assert os.path.isfile(path), "lightmap file missing on disk: %s" % path
    group("lightmap: image + .exr on disk", check_lightmap)

    # --- summary ------------------------------------------------------------------
    print("\n== bl_smoke summary ==")
    for name, ok in _results:
        print(("PASS  " if ok else "FAIL  ") + name)
    failed = [name for name, ok in _results if not ok]
    if failed:
        print("%d/%d assertion groups FAILED" % (len(failed), len(_results)))
        sys.exit(1)
    print("all %d assertion groups passed" % len(_results))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
