# Re-bake guard + presets checks.
#
#  A) §5 stage 0: "source-mesh edits change the hash and re-enable processing"
#     -> a sculpt-style edit that preserves vert/poly counts must clear the
#        already_baked guard.
#  B) §8: factory presets Props 2k / Hero 4k / Mobile 1k exist, live in the
#     extension user dir as JSON, round-trip through save/load, survive a
#     rewrite (the module-level FACTORY dict must never be mutated), and
#     Mobile 1k selects letter-suffix naming.
#
# Run:
#   blender --background --factory-startup --python-exit-code 1 \
#       --python tests/bl_guard_presets.py
#
# Prints PASS/FAIL per check; exits 1 on any failure.

import json
import os
import sys
import tempfile
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import bpy

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


def build():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.8)
    s = bpy.context.active_object
    s.name = "GuardSphere"
    m = bpy.data.materials.new("GuardMat")
    m.use_nodes = True
    s.data.materials.append(m)
    bpy.ops.object.select_all(action="DESELECT")
    s.select_set(True)
    bpy.context.view_layer.objects.active = s
    return s


def main():
    import fastbake
    fastbake.register()
    from fastbake.pipeline import driver, preflight
    from fastbake import presets

    outdir = tempfile.mkdtemp(prefix="fastbake_guard_")
    src = build()
    fb = bpy.context.scene.fastbake
    fb.quads = 500
    fb.tex_size = "512"
    fb.advanced.output_dir = outdir

    ctx, err = preflight.make_context(bpy.context)
    assert not err, err
    driver.launch(bpy.context, driver.build_bake_stages(), ctx)
    print("[guard] first bake produced: %s" % ctx.result_name)

    def reselect():
        bpy.ops.object.select_all(action="DESELECT")
        src.hide_set(False)
        src.hide_viewport = False
        src.select_set(True)
        bpy.context.view_layer.objects.active = src

    def guard_fires_unchanged():
        reselect()
        c, e = preflight.make_context(bpy.context)
        assert not e, e
        assert c.flags.get("already_baked"), "guard did NOT fire on an unchanged re-run"
        print("[guard] unchanged  -> already_baked = %r" % c.flags["already_baked"])
    group("A1 guard fires on unchanged selection", guard_fires_unchanged)

    def guard_clears_after_edit():
        # move every vertex: a sculpt-style edit. Vertex and polygon counts
        # are identical, only coordinates change.
        nverts, npolys = len(src.data.vertices), len(src.data.polygons)
        for v in src.data.vertices:
            v.co.x *= 1.35
            v.co.z += 0.22
        src.data.update()
        assert len(src.data.vertices) == nverts and len(src.data.polygons) == npolys
        reselect()
        c, e = preflight.make_context(bpy.context)
        assert not e, e
        flag = c.flags.get("already_baked")
        print("[guard] after edit -> already_baked = %r" % flag)
        assert not flag, ("guard STILL fires after a shape-changing edit "
                          "(hash ignores vertex positions)")
    group("A2 guard clears after a shape-changing edit", guard_clears_after_edit)

    # --- B: presets -------------------------------------------------------------
    def factory_presets_exist():
        names = presets.list_presets()
        print("[presets] found: %s" % names)
        for want in ("Props 2k", "Hero 4k", "Mobile 1k"):
            assert want in names, "factory preset %r missing (have %s)" % (want, names)
    group("B1 three factory presets present", factory_presets_exist)

    def presets_are_json_in_extension_dir():
        d = presets.preset_dir()
        print("[presets] dir: %s" % d)
        files = sorted(f for f in os.listdir(d) if f.endswith(".json"))
        assert files, "no .json preset files in %s" % d
        with open(os.path.join(d, files[0])) as fh:
            blob = json.load(fh)
        assert "fastbake_preset" in blob, "preset file lacks the schema marker: %s" % blob.keys()
        print("[presets] %s -> keys %s" % (files[0], sorted(blob.keys())))
    group("B2 presets stored as JSON with schema marker", presets_are_json_in_extension_dir)

    def mobile_preset_uses_letters():
        presets.load_preset("Mobile 1k", bpy.context.scene.fastbake)
        s = bpy.context.scene.fastbake
        print("[presets] Mobile 1k -> quads=%d tex=%s style=%s"
              % (s.quads, s.tex_size, s.advanced.naming_style))
        assert s.advanced.naming_style == "LETTERS", \
            "Mobile 1k should select letter suffixes, got %s" % s.advanced.naming_style
        assert s.tex_size == "1024"
    group("B3 Mobile 1k selects letter suffixes", mobile_preset_uses_letters)

    def roundtrip():
        s = bpy.context.scene.fastbake
        s.quads = 4242
        s.advanced.margin_px = 23
        presets.save_preset("RoundTrip Test", s)
        s.quads = 100
        s.advanced.margin_px = 1
        presets.load_preset("RoundTrip Test", s)
        assert s.quads == 4242, "quads did not survive the round-trip: %d" % s.quads
        assert s.advanced.margin_px == 23, "margin did not survive: %d" % s.advanced.margin_px
        os.remove(presets._path("RoundTrip Test"))
        assert "RoundTrip Test" not in presets.list_presets(), "preset file not removed"
    group("B4 user preset save/load round-trip", roundtrip)

    def factory_rewrite_keeps_letters():
        # Delete the Mobile 1k file and let ensure_factory_presets() rewrite it,
        # exactly as a re-enable of the add-on would. FACTORY must not have been
        # mutated by the first call.
        os.remove(presets._path("Mobile 1k"))
        presets.ensure_factory_presets()
        with open(presets._path("Mobile 1k")) as fh:
            blob = json.load(fh)
        print("[presets] rewritten Mobile 1k -> %s" % blob)
        assert blob.get("advanced", {}).get("naming_style") == "LETTERS", \
            ("rewritten Mobile 1k lost naming_style=LETTERS: %s — "
             "ensure_factory_presets() mutated the module-level FACTORY dict"
             % blob.get("advanced"))
    group("B5 factory preset survives a rewrite (FACTORY not mutated)",
          factory_rewrite_keeps_letters)

    def name_sanitising_is_symmetric():
        # A name with path/OS-hostile characters must come back from
        # list_presets() exactly as the name that load_preset() accepts —
        # sanitise once, in one direction only.
        raw = "Weird/Name: 2k?"
        s = bpy.context.scene.fastbake
        s.quads = 1234
        try:                                      # leftover from an aborted run
            os.remove(presets._path(raw))
        except OSError:
            pass
        before = set(presets.list_presets())      # factory presets may grow
        presets.save_preset(raw, s)
        listed = sorted(set(presets.list_presets()) - before)
        assert listed, "saved preset %r does not appear in list_presets()" % raw
        assert len(listed) == 1, "saving one preset added %s" % listed
        name = listed[0]
        print("[presets] %r stored and listed as %r" % (raw, name))
        s.quads = 99
        presets.load_preset(name, s)
        assert s.quads == 1234, \
            "listed name %r does not load back (asymmetric sanitising)" % name
        os.remove(presets._path(name))
    group("B6 preset names sanitise symmetrically", name_sanitising_is_symmetric)

    print("\n== guard/presets summary ==")
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
