# Headless check for the source-alpha path (plan §2 ruling: "the original
# alpha wins the alpha channel" — no hidden data loss).
#
# Run:
#   /Applications/Blender.app/Contents/MacOS/Blender --background \
#       --factory-startup --python tests/bl_alpha.py
#
# Bakes one source whose Principled Alpha is 0.5 and asserts:
#   * the saved color PNG's alpha channel is ~0.5 (the SOURCE alpha), not the
#     bake coverage mask (which would be 1.0 inside the islands),
#   * roughness was saved as its own file (the §2 override beats the
#     "Roughness to alpha" checkbox, which is left ON on purpose),
#   * the transient "alpha" map is gone from ctx.images / ctx.files,
#   * the result material wires color Alpha -> Principled Alpha,
#   * the result keeps the fixed-diagonal Triangulate modifier (decision 5).

import os
import sys
import tempfile
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bpy
import numpy as np

SRC_ALPHA = 0.5
TARGET_QUADS = 600

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


def build_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1.0)
    src = bpy.context.active_object
    src.name = "AlphaBlob"

    mat = bpy.data.materials.new("AlphaCard")
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (0.2, 0.6, 0.9, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Alpha"].default_value = SRC_ALPHA
    src.data.materials.append(mat)

    bpy.ops.object.select_all(action="DESELECT")
    src.select_set(True)
    bpy.context.view_layer.objects.active = src
    return src


def read_png_pixels(path):
    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf


def main():
    import fastbake
    fastbake.register()
    from fastbake.pipeline import driver, preflight

    build_scene()
    outdir = tempfile.mkdtemp(prefix="fastbake_alpha_")
    fb = bpy.context.scene.fastbake
    fb.quads = TARGET_QUADS
    fb.tex_size = "512"
    fb.roughness_to_alpha = True          # the §2 ruling must override this
    fb.advanced.ssaa = "1"                # keep the run short
    fb.advanced.output_dir = outdir
    print("[alpha_bake] output dir: " + outdir)

    ctx, err = preflight.make_context(bpy.context)
    assert not err, "make_context failed: %s" % err
    assert driver.launch(bpy.context, driver.build_bake_stages(), ctx), \
        "driver.launch refused to start"

    def check_flags():
        assert ctx.flags.get("source_uses_alpha"), "source_uses_alpha not detected"
        assert ctx.flags.get("source_alpha_packed"), \
            "source alpha was never packed into the color image"
        assert not ctx.flags.get("roughness_packed"), \
            "roughness must stay separate when the source drives alpha"
        assert "alpha" not in ctx.images, \
            "transient alpha map left in ctx.images: %s" % sorted(ctx.images)
        assert "alpha" not in ctx.files, "alpha map was saved to disk"
    group("flags: source alpha packed, roughness separate", check_flags)

    def check_roughness_file():
        path = ctx.files.get("roughness")
        assert path, "roughness was not saved as its own file"
        assert os.path.isfile(path), "roughness file missing on disk: %s" % path
        assert path != ctx.files.get("color"), "roughness and color share a file"
    group("roughness saved as a separate PNG", check_roughness_file)

    def check_color_alpha():
        path = ctx.files.get("color")
        assert path and os.path.isfile(path), "color PNG missing: %r" % path
        buf = read_png_pixels(path)
        a = buf[3::4]
        covered = a > 0.02                       # everything the bake touched
        assert covered.mean() > 0.2, \
            "suspiciously little coverage (%.3f) — bake may have failed" % covered.mean()
        hit = np.abs(a - SRC_ALPHA) < 0.03
        print("[alpha_bake] alpha: max %.4f, mean over covered %.4f, "
              "at %.2f: %.1f%% of texels"
              % (a.max(), a[covered].mean(), SRC_ALPHA, 100.0 * hit.mean()))
        assert a.max() < SRC_ALPHA + 0.03, (
            "alpha channel reaches %.3f — that is the bake COVERAGE mask, not "
            "the source alpha" % a.max())
        assert hit.mean() > 0.2, \
            "only %.1f%% of texels carry the source alpha" % (100.0 * hit.mean())
        assert abs(a[covered].mean() - SRC_ALPHA) < 0.03, \
            "covered alpha averages %.4f, expected ~%.2f" % (a[covered].mean(), SRC_ALPHA)
    group("color PNG alpha == source alpha (not the coverage mask)", check_color_alpha)

    def check_material_wiring():
        mat = bpy.data.materials.get((ctx.material_name, None))
        assert mat is not None, "result material %r missing" % ctx.material_name
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        alpha_in = bsdf.inputs["Alpha"]
        assert alpha_in.is_linked, "Principled Alpha is not connected"
        from_node = alpha_in.links[0].from_node
        assert from_node.type == "TEX_IMAGE", \
            "Alpha driven by %s, expected the color texture" % from_node.type
        assert from_node.image.name == ctx.images["color"], \
            "Alpha driven by %s, expected %s" % (from_node.image.name,
                                                 ctx.images["color"])
        assert alpha_in.links[0].from_socket.name == "Alpha", \
            "Alpha driven by the %s socket" % alpha_in.links[0].from_socket.name
        method = getattr(mat, "surface_render_method", None)
        if method is not None:
            assert method in ("BLENDED", "DITHERED"), \
                "surface render method left at %r" % method
        rough_in = bsdf.inputs["Roughness"]
        assert rough_in.is_linked, "Roughness is not connected"
        assert rough_in.links[0].from_node.image.name == ctx.images["roughness"], \
            "Roughness should read the separate roughness texture"
    group("material: color Alpha -> Principled Alpha, roughness separate",
          check_material_wiring)

    def check_triangulate_kept():
        result = bpy.data.objects[ctx.result_name]
        names = [m.name for m in result.modifiers]
        assert "FB_Triangulate" in names, \
            "Triangulate modifier not kept for export (have: %s)" % names
        mod = result.modifiers["FB_Triangulate"]
        assert mod.quad_method == "FIXED", \
            "Triangulate quad_method is %r, must stay FIXED" % mod.quad_method
        assert any("Triangulate" in n for n in ctx.notices), \
            "no notice about the kept Triangulate modifier: %s" % ctx.notices
    group("result keeps the fixed-diagonal Triangulate + notice",
          check_triangulate_kept)

    def check_sources_restored():
        src = bpy.data.objects.get("AlphaBlob")
        assert src is not None, "source object vanished"
        mat = src.material_slots[0].material
        kinds = sorted(n.type for n in mat.node_tree.nodes)
        assert "EMISSION" not in kinds, \
            "temporary Emission node left in the source material: %s" % kinds
        assert "VALUE" not in kinds, \
            "temporary Value node left in the source material: %s" % kinds
        out = mat.node_tree.get_output_node("CYCLES")
        assert out.inputs["Surface"].is_linked, "source Surface left unconnected"
        assert out.inputs["Surface"].links[0].from_node.type == "BSDF_PRINCIPLED", \
            "source Surface no longer driven by its Principled BSDF"
        assert not bpy.data.materials.get(("FB_alpha_opaque", None)), \
            "temporary opaque material left behind"
        assert not bpy.data.images.get(("FB_alpha_tmp", None)), \
            "transient alpha image left behind"
    group("source material rewire fully restored", check_sources_restored)

    print("\n== alpha_bake summary ==")
    for name, ok in _results:
        print(("PASS  " if ok else "FAIL  ") + name)
    failed = [name for name, ok in _results if not ok]
    if failed:
        print("%d/%d checks FAILED" % (len(failed), len(_results)))
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
