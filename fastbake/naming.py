# FastBake naming engine. Pure Python — no bpy imports, unit-testable.
#
# Conventions (see dev/CONTRACTS.md and plan §6):
#   mesh      <Base>_01, _02 ...            (always numbered, starts at _01)
#   textures  T_<Base>_color                (run 0 = unnumbered, run 1 = _01 ...)
#   material  M_<Base>                      (same run counter as textures)
# The run counter is shared by all textures + material of one bake so a
# re-bake never yields a mixed set.
#
# Naming is TEMPLATE-DRIVEN (plan §6 "naming lives in an editable template"):
# the strings above are the default templates, editable through
# FastBakeAdvanced.naming_template / .material_template. Placeholders:
#   {base}  asset base name        {map}  map suffix (style table below)
#   {run}   "" for run 0, else _NN
# Unknown placeholders expand to nothing and a malformed template falls back
# to the default, so a bad string can never abort a bake.

import re

_TRAIL = re.compile(r"^(.*?)(?:[._]\d{2,3})+$")

MAP_SUFFIX = {
    "VERBOSE": {"color": "color", "normal": "normal",
                "roughness": "roughness", "lightmap": "lightmap"},
    "LETTERS": {"color": "D", "normal": "N",
                "roughness": "R", "lightmap": "L"},
}

DEFAULT_TEX_TEMPLATE = "T_{base}_{map}{run}"
DEFAULT_MAT_TEMPLATE = "M_{base}{run}"

_RUN_TOKEN = "{run}"
# a run suffix inside a name: _01 .. _999 (4+ digits are someone else's)
_RUN_PATTERN = r"(?:_(\d{2,3}))?"


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


# --- template expansion --------------------------------------------------------

class _Fields(dict):
    """format_map() mapping that renders unknown placeholders as nothing."""

    def __missing__(self, key):
        return ""


def _fields(base, map_suffix, run):
    return _Fields(base=base, map=map_suffix, run=_num(run) if run else "")


def _expand(template, default, base, map_suffix, run):
    fields = _fields(base, map_suffix, run)
    for tpl in (template, default):
        if not tpl:
            continue
        try:
            out = tpl.format_map(fields)
        except (IndexError, KeyError, ValueError, TypeError, AttributeError):
            continue  # malformed template ({0}, unbalanced braces, ...)
        out = out.strip()
        if out:
            return out
    return default.format_map(fields)


def map_suffix(map_key, style="VERBOSE"):
    """Suffix for one map key in the given style. KeyError on unknown keys."""
    return MAP_SUFFIX.get(style, MAP_SUFFIX["VERBOSE"])[map_key]


def tex_name(base, map_key, run, style="VERBOSE", template=None):
    """Texture datablock name. `template` defaults to T_{base}_{map}{run}."""
    return _expand(template, DEFAULT_TEX_TEMPLATE, base,
                   map_suffix(map_key, style), run)


def mat_name(base, run, template=None):
    """Material datablock name. `template` defaults to M_{base}{run}."""
    return _expand(template, DEFAULT_MAT_TEMPLATE, base, "", run)


def _run_regex(template, default, base, map_suffix_text):
    """Regex matching every name the template can produce for any run.

    Splitting on {run} keeps the rest of the template literal, so a custom
    template ('{base}_{map}{run}_bake', 'T_{base}{run}_{map}', ...) is
    recognised by next_run exactly where it puts its number.
    """
    for tpl in (template, default):
        if not tpl:
            continue
        # _expand() strips the rendered name, so strip the template here too
        # or a pasted " T_{base}_{map}{run} " would never match its own output
        pieces = tpl.strip().split(_RUN_TOKEN)
        fields = _fields(base, map_suffix_text, 0)
        try:
            literals = [p.format_map(fields) for p in pieces]
        except (IndexError, KeyError, ValueError, TypeError, AttributeError):
            continue
        if not any(literals):
            continue
        body = _RUN_PATTERN.join(re.escape(p) for p in literals)
        return re.compile("^" + body + "$")
    return re.compile(r"^(?!)")  # never matches


def next_run(base, existing, style="VERBOSE",
             maps=("color", "normal", "roughness", "lightmap"),
             template=None, material_template=None, include_material=True):
    """Next free shared run counter given existing texture/material names.

    `existing` is any iterable of names (blend images, materials and/or
    output-directory file stems). Run 0 (unnumbered) is used first; if any
    artifact of run N exists the whole set moves to N+1. Only names the
    ACTIVE templates can produce count.

    `maps` / `include_material` narrow the scan: a standalone lightmap counts
    only existing lightmaps, so the sibling maps of a finished bake cannot
    push its first lightmap to _01 (plan §6 table).
    """
    patterns = []
    for key in maps:
        try:
            suffix = map_suffix(key, style)
        except KeyError:
            continue  # map key from another version
        patterns.append(_run_regex(template, DEFAULT_TEX_TEMPLATE, base, suffix))
    if include_material:
        patterns.append(_run_regex(material_template, DEFAULT_MAT_TEMPLATE, base, ""))

    used = -1
    for name in existing:
        for pattern in patterns:
            m = pattern.match(name)
            if m is None:
                continue
            numbers = [int(g) for g in m.groups() if g]
            used = max(used, max(numbers) if numbers else 0)
            break
    return used + 1


def strip_ext(filename):
    """'T_Rock_color.png' -> 'T_Rock_color'"""
    i = filename.rfind(".")
    return filename[:i] if i > 0 else filename
