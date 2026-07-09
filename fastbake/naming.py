# FastBake naming engine. Pure Python — no bpy imports, unit-testable.
#
# Conventions (see dev/CONTRACTS.md):
#   mesh      <Base>_01, _02 ...            (always numbered, starts at _01)
#   textures  T_<Base>_color                (run 0 = unnumbered, run 1 = _01 ...)
#   material  M_<Base>                      (same run counter as textures)
# The run counter is shared by all textures + material of one bake so a
# re-bake never yields a mixed set.

import re

_TRAIL = re.compile(r"^(.*?)(?:[._]\d{2,3})+$")

MAP_SUFFIX = {
    "VERBOSE": {"color": "color", "normal": "normal",
                "roughness": "roughness", "lightmap": "lightmap"},
    "LETTERS": {"color": "D", "normal": "N",
                "roughness": "R", "lightmap": "L"},
}


def base_name(name):
    """Strip trailing Blender .NNN and FastBake _NN suffixes (repeatedly).

    'Cube' -> 'Cube';  'Cube_01' -> 'Cube';  'Cube.001' -> 'Cube';
    'Cube_01.003' -> 'Cube';  'Rock_2020' -> 'Rock_2020' (4+ digits kept).
    """
    m = _TRAIL.match(name)
    return m.group(1) if m and m.group(1) else name


def _num(n):
    return "_%02d" % n


def next_mesh_name(base, existing):
    """Smallest free '<base>_NN' (NN >= 01) not present in `existing`."""
    existing = set(existing)
    n = 1
    while base + _num(n) in existing:
        n += 1
    return base + _num(n)


def tex_name(base, map_key, run, style="VERBOSE"):
    suffix = MAP_SUFFIX.get(style, MAP_SUFFIX["VERBOSE"])[map_key]
    return "T_%s_%s%s" % (base, suffix, _num(run) if run else "")


def mat_name(base, run):
    return "M_%s%s" % (base, _num(run) if run else "")


def _tex_stem(base, map_key, style):
    return "T_%s_%s" % (base, MAP_SUFFIX.get(style, MAP_SUFFIX["VERBOSE"])[map_key])


def next_run(base, existing, style="VERBOSE", maps=("color", "normal", "roughness", "lightmap")):
    """Next free shared run counter given existing texture/material names.

    `existing` is any iterable of names (blend images, materials and/or
    output-directory file stems). Run 0 (unnumbered) is used first; if any
    artifact of run N exists the whole set moves to N+1.
    """
    existing = {e for e in existing}
    used = -1
    stems = [_tex_stem(base, m, style) for m in maps] + ["M_" + base]
    for name in existing:
        for stem in stems:
            if name == stem:
                used = max(used, 0)
            elif name.startswith(stem + "_"):
                tail = name[len(stem) + 1:]
                if tail.isdigit() and len(tail) in (2, 3):
                    used = max(used, int(tail))
    return used + 1


def strip_ext(filename):
    """'T_Rock_color.png' -> 'T_Rock_color'"""
    i = filename.rfind(".")
    return filename[:i] if i > 0 else filename
