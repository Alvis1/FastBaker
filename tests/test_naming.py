# Pure-python tests for the naming engine. Run: python3 tests/test_naming.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# naming.py has no bpy imports — import the module file directly
import importlib.util

spec = importlib.util.spec_from_file_location(
    "naming", os.path.join(os.path.dirname(__file__), "..", "fastbake", "naming.py"))
naming = importlib.util.module_from_spec(spec)
spec.loader.exec_module(naming)


def test_base_name():
    assert naming.base_name("Cube") == "Cube"
    assert naming.base_name("Cube_01") == "Cube"
    assert naming.base_name("Cube.001") == "Cube"
    assert naming.base_name("Cube_01.003") == "Cube"
    assert naming.base_name("Rock_2020") == "Rock_2020"      # 4 digits: not ours
    assert naming.base_name("Tree_09_02") == "Tree"          # repeated strip
    assert naming.base_name("_01") == "_01"                  # never empty


def test_next_mesh_name():
    assert naming.next_mesh_name("Cube", []) == "Cube_01"
    assert naming.next_mesh_name("Cube", ["Cube_01"]) == "Cube_02"
    assert naming.next_mesh_name("Cube", ["Cube_01", "Cube_02"]) == "Cube_03"
    assert naming.next_mesh_name("Cube", ["Cube_02"]) == "Cube_01"  # fills gaps


def test_tex_and_mat_names():
    assert naming.tex_name("Rock", "color", 0) == "T_Rock_color"
    assert naming.tex_name("Rock", "color", 1) == "T_Rock_color_01"
    assert naming.tex_name("Rock", "normal", 12) == "T_Rock_normal_12"
    assert naming.tex_name("Rock", "color", 0, "LETTERS") == "T_Rock_D"
    assert naming.tex_name("Rock", "normal", 2, "LETTERS") == "T_Rock_N_02"
    assert naming.mat_name("Rock", 0) == "M_Rock"
    assert naming.mat_name("Rock", 3) == "M_Rock_03"


def test_next_run():
    assert naming.next_run("Rock", []) == 0
    assert naming.next_run("Rock", ["T_Rock_color"]) == 1
    assert naming.next_run("Rock", ["T_Rock_color", "T_Rock_normal"]) == 1
    assert naming.next_run("Rock", ["T_Rock_color", "T_Rock_color_01"]) == 2
    assert naming.next_run("Rock", ["M_Rock_04"]) == 5
    # unrelated names don't count
    assert naming.next_run("Rock", ["T_Stone_color", "M_Stone"]) == 0
    # file stems from the output dir work the same
    assert naming.next_run("Rock", [naming.strip_ext("T_Rock_color_02.png")]) == 3
    # letter style
    assert naming.next_run("Rock", ["T_Rock_D"], style="LETTERS") == 1


def test_template_defaults_match_hardcoded_scheme():
    # the default templates must reproduce the historical names byte for byte
    assert naming.DEFAULT_TEX_TEMPLATE == "T_{base}_{map}{run}"
    assert naming.DEFAULT_MAT_TEMPLATE == "M_{base}{run}"
    assert naming.tex_name("Rock", "color", 0, "VERBOSE",
                           naming.DEFAULT_TEX_TEMPLATE) == "T_Rock_color"
    assert naming.tex_name("Rock", "normal", 7, "VERBOSE",
                           naming.DEFAULT_TEX_TEMPLATE) == "T_Rock_normal_07"
    assert naming.mat_name("Rock", 0, naming.DEFAULT_MAT_TEMPLATE) == "M_Rock"
    assert naming.mat_name("Rock", 7, naming.DEFAULT_MAT_TEMPLATE) == "M_Rock_07"


def test_custom_templates():
    tpl = "{base}_{map}{run}_bake"
    assert naming.tex_name("Rock", "color", 0, "VERBOSE", tpl) == "Rock_color_bake"
    assert naming.tex_name("Rock", "color", 2, "VERBOSE", tpl) == "Rock_color_02_bake"
    assert naming.tex_name("Rock", "color", 0, "LETTERS", tpl) == "Rock_D_bake"
    # run in the middle, material template without {map}
    assert naming.tex_name("Rock", "normal", 3, "VERBOSE",
                           "T_{base}{run}_{map}") == "T_Rock_03_normal"
    assert naming.mat_name("Rock", 3, "MI_{base}{run}") == "MI_Rock_03"
    # unknown placeholders vanish instead of raising
    assert naming.tex_name("Rock", "color", 0, "VERBOSE",
                           "{base}_{map}{unknown}") == "Rock_color"
    # a broken template falls back to the default, never to an empty name
    assert naming.tex_name("Rock", "color", 0, "VERBOSE", "{0}") == "T_Rock_color"
    assert naming.tex_name("Rock", "color", 0, "VERBOSE", "") == "T_Rock_color"
    assert naming.mat_name("Rock", 0, "{unknown}") == "M_Rock"


def test_next_run_follows_the_template():
    tpl = "{base}_{map}{run}_bake"
    mat = "MI_{base}{run}"
    kw = dict(template=tpl, material_template=mat)
    assert naming.next_run("Rock", ["Rock_color_bake"], **kw) == 1
    assert naming.next_run("Rock", ["Rock_color_02_bake"], **kw) == 3
    assert naming.next_run("Rock", ["MI_Rock_04"], **kw) == 5
    # names of the DEFAULT scheme do not count when a custom one is active
    assert naming.next_run("Rock", ["T_Rock_color", "M_Rock"], **kw) == 0
    # ... and vice versa
    assert naming.next_run("Rock", ["Rock_color_bake"]) == 0
    # 4-digit tails are somebody else's numbering
    assert naming.next_run("Rock", ["T_Rock_color_2020"]) == 0


def test_next_run_scan_can_be_narrowed():
    # a standalone lightmap must not be bumped by the maps of a finished bake
    existing = ["T_Rock_color", "T_Rock_normal", "M_Rock"]
    assert naming.next_run("Rock", existing) == 1
    assert naming.next_run("Rock", existing, maps=("lightmap",),
                           include_material=False) == 0
    assert naming.next_run("Rock", existing + ["T_Rock_lightmap"],
                           maps=("lightmap",), include_material=False) == 1


def test_strip_ext():
    assert naming.strip_ext("T_Rock_color.png") == "T_Rock_color"
    assert naming.strip_ext("noext") == "noext"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError:
                import traceback
                traceback.print_exc()
                fails += 1
                print("FAIL", name)
    sys.exit(1 if fails else 0)
