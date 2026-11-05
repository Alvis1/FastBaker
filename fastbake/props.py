# Settings model. Scene-level (saves with the .blend).
#
# The JSON preset layer (presets.py) serializes this group through
# settings_to_dict / dict_to_settings — bl_rna introspection only, no exec
# (plan §8).

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

from . import compat
from .naming import DEFAULT_MAT_TEMPLATE, DEFAULT_TEX_TEMPLATE

TEX_SIZES = [(s, s, "") for s in ("512", "1024", "2048", "4096", "8192")]
LM_SIZES = [(s, s, "") for s in ("256", "512", "1024", "2048", "4096")]


class FastBakeAdvanced(bpy.types.PropertyGroup):
    output_dir: StringProperty(
        name="Output", subtype="DIR_PATH", default="//textures/",
        options=compat.path_prop_options(),
        description="Directory where baked textures are saved")
    margin_px: IntProperty(name="Margin", default=16, min=0, max=256,
                           description="Bake margin in pixels per 1K of resolution")
    ssaa: EnumProperty(
        name="Anti-aliasing", default="2",
        items=[("1", "Off", "Bake at target resolution"),
               ("2", "2x supersampling", "Bake at 2x and downscale")])
    normal_y: EnumProperty(
        name="Normal Y", default="OPENGL",
        items=[("OPENGL", "OpenGL (Y+)", "Blender, Unity, glTF"),
               ("DIRECTX", "DirectX (Y-)", "Unreal-style green channel")])
    engine: EnumProperty(
        name="Retopo engine", default="AUTO",
        items=[("AUTO", "Auto", "AutoRemesher when available, else QuadriFlow"),
               ("AUTOREMESHER", "AutoRemesher",
                "Needs the AutoRemesher extension enabled, a platform build of "
                "FastBake, or the AutoRemesher program"),
               ("QUADRIFLOW", "QuadriFlow", "Blender built-in")])
    merge_detail: EnumProperty(
        name="Merge detail", default="MED",
        items=[("LOW", "Low", "Voxel = diagonal/150"),
               ("MED", "Medium", "Voxel = diagonal/300"),
               ("HIGH", "High", "Voxel = diagonal/600")])
    merge_mode: EnumProperty(
        name="Merge mode", default="AUTO",
        items=[("AUTO", "Auto",
                "Boolean union of clean shells, voxel remesh otherwise"),
               ("OUTER_SHELL", "Outer shell",
                "Keep only the exterior hull: seal openings narrower than the "
                "probe and discard interiors (rooms, cavities, hidden parts)")])
    shell_probe: FloatProperty(
        name="Probe radius", default=3.0, min=0.1, max=25.0, subtype="PERCENTAGE",
        description="Outer shell: openings narrower than twice this radius "
                    "(as % of the bounding diagonal) are sealed")
    shell_max_bridged: FloatProperty(
        name="Max bridged", default=15.0, min=0.0, max=100.0, subtype="PERCENTAGE",
        description="Outer shell: cancel the run when more than this share of "
                    "the hull's area would be invented surface")
    model_type: EnumProperty(
        name="Surface type", default="ORGANIC",
        items=[("ORGANIC", "Organic", ""), ("HARDSURFACE", "Hard surface", "")])
    sharp_edge: FloatProperty(name="Sharp edge angle", default=90.0, min=30.0, max=180.0)
    protect_view: BoolProperty(
        name="Protect view", default=False,
        description="Penalize seams on faces visible from the current viewport")
    naming_style: EnumProperty(
        name="Suffix style", default="VERBOSE",
        items=[("VERBOSE", "Verbose (_color, _normal)", ""),
               ("LETTERS", "Unreal letters (_D, _N)", "")])
    naming_template: StringProperty(
        name="Texture name", default=DEFAULT_TEX_TEMPLATE,
        description="Template for baked texture names. "
                    "{base} asset name, {map} suffix (see Suffix style), "
                    "{run} bake number (empty on the first bake)")
    material_template: StringProperty(
        name="Material name", default=DEFAULT_MAT_TEMPLATE,
        description="Template for the baked material name. "
                    "{base} asset name, {run} bake number")
    # Lightmap settings (edited in the pre-bake dialog, persisted here)
    lm_size: EnumProperty(name="Lightmap size", items=LM_SIZES, default="1024")
    lm_samples: IntProperty(name="Samples", default=64, min=1, max=4096)
    lm_gamma: FloatProperty(name="Gamma", default=1.0, min=0.2, max=5.0,
                            description="Post-bake gamma; 1.0 keeps the lightmap linear")
    lm_denoise: BoolProperty(name="Denoise", default=True)


class FastBakeSettings(bpy.types.PropertyGroup):
    quads: IntProperty(
        name="Quads", default=5000, min=100, soft_max=100000, max=2000000,
        description="Target quad count for the retopologized mesh")
    tex_size: EnumProperty(name="Texture", items=TEX_SIZES, default="2048",
                           description="Baked texture resolution")
    roughness_to_alpha: BoolProperty(
        name="Roughness to alpha", default=True,
        description="Pack the roughness map into the color texture's alpha channel")
    review_stages: BoolProperty(
        name="Review each stage", default=False,
        description="Pause after every pipeline stage to review the result")
    advanced: PointerProperty(type=FastBakeAdvanced)
    show_advanced: BoolProperty(default=False)
    # set once per scene by the load_post handler that seeds machine-level
    # defaults from the add-on preferences (plan §8); never user-visible
    initialized: BoolProperty(default=False, options={"HIDDEN"})


# --- snapshot ---------------------------------------------------------------

_SNAP_KEYS = ("quads", "tex_size", "roughness_to_alpha", "review_stages")
_SNAP_ADV = ("output_dir", "margin_px", "ssaa", "normal_y", "engine",
             "merge_detail", "merge_mode", "shell_probe", "shell_max_bridged",
             "model_type", "sharp_edge", "protect_view",
             "naming_style", "naming_template", "material_template",
             "lm_size", "lm_samples", "lm_gamma", "lm_denoise")

# plan §8 preset schema sugar: a preset may carry its naming block and its
# lightmap block nested instead of flat. Both directions are accepted so a
# hand-written preset from the documented schema applies in full.
_NAMING_BLOCK = {"template": "naming_template",
                 "material": "material_template",
                 "style": "naming_style"}
_LIGHTMAP_BLOCK = {"size": "lm_size", "samples": "lm_samples",
                   "gamma": "lm_gamma", "denoise": "lm_denoise"}
# keys of the preset file that are not settings
_SCHEMA_KEYS = ("fastbake_preset", "name")


def settings_to_dict(settings):
    d = {k: getattr(settings, k) for k in _SNAP_KEYS}
    d["advanced"] = {k: getattr(settings.advanced, k) for k in _SNAP_ADV}
    return d


def _rna_default(group_type, key):
    prop = group_type.bl_rna.properties.get(key)
    if prop is None:
        raise KeyError(key)
    return prop.default


def default_settings_dict():
    """settings_to_dict() of a pristine settings group, read from the RNA.

    Used to write factory presets COMPLETE (every key present), so loading a
    preset can never leave an unlisted setting at the previous preset's value.
    """
    d = {k: _rna_default(FastBakeSettings, k) for k in _SNAP_KEYS}
    d["advanced"] = {k: _rna_default(FastBakeAdvanced, k) for k in _SNAP_ADV}
    return d


def _coerce(group, key, value):
    """Convert a JSON value to what this property accepts, or raise."""
    prop = group.bl_rna.properties.get(key)
    if prop is None:
        raise KeyError(key)
    kind = prop.type
    if kind == "ENUM":
        if isinstance(value, bool):
            raise ValueError("bool for an enum")
        text = ("%d" % value) if isinstance(value, int) else str(value)
        identifiers = [i.identifier for i in prop.enum_items]
        if text in identifiers:
            return text
        for ident in identifiers:            # "verbose" -> "VERBOSE"
            if ident.lower() == text.lower():
                return ident
        raise ValueError("%r is not one of %s" % (value, identifiers))
    if kind == "BOOLEAN":
        if isinstance(value, str):
            raise ValueError("string for a boolean")
        return bool(value)
    if kind == "INT":
        if isinstance(value, bool):
            raise ValueError("bool for an int")
        return int(value)
    if kind == "FLOAT":
        if isinstance(value, bool):
            raise ValueError("bool for a float")
        return float(value)
    if kind == "STRING":
        if not isinstance(value, str):
            raise ValueError("%r is not a string" % (value,))
        return value
    raise ValueError("unsupported property type %s" % kind)


def _apply(group, key, value, rejected, label):
    try:
        setattr(group, key, _coerce(group, key, value))
    except (KeyError, TypeError, ValueError) as exc:
        # tolerate presets from other versions, but never silently: the
        # caller reports what did not apply
        rejected.append("%s (%s)" % (label, exc))


def dict_to_settings(d, settings):
    """Apply a preset dict; return the list of keys that did NOT apply.

    Values are coerced to the property's type first (an int 2048 for the
    tex_size enum is the documented schema, plan §8), so a hand-written
    preset applies in full instead of partially and silently.
    """
    rejected = []
    for key, value in d.items():
        if key in _SCHEMA_KEYS or key in ("advanced", "naming"):
            continue
        if key in _SNAP_KEYS:
            _apply(settings, key, value, rejected, key)
        else:
            rejected.append("%s (unknown setting)" % key)

    adv = d.get("advanced")
    if isinstance(adv, dict):
        for key, value in adv.items():
            if key in _SNAP_ADV:
                _apply(settings.advanced, key, value, rejected, "advanced." + key)
            elif key == "lightmap" and isinstance(value, dict):
                for sub, target in _LIGHTMAP_BLOCK.items():
                    if sub in value:
                        _apply(settings.advanced, target, value[sub], rejected,
                               "advanced.lightmap." + sub)
                for sub in value:
                    if sub not in _LIGHTMAP_BLOCK:
                        rejected.append("advanced.lightmap.%s (unknown setting)" % sub)
            else:
                rejected.append("advanced.%s (unknown setting)" % key)
    elif adv is not None:
        rejected.append("advanced (not a settings block)")

    naming_block = d.get("naming")
    if isinstance(naming_block, dict):
        for key, value in naming_block.items():
            target = _NAMING_BLOCK.get(key)
            if target is None:
                rejected.append("naming.%s (unknown setting)" % key)
            else:
                _apply(settings.advanced, target, value, rejected, "naming." + key)
    elif naming_block is not None:
        rejected.append("naming (not a settings block)")
    return rejected


def naming_block(settings_dict):
    """The plan §8 'naming' block mirroring the advanced naming settings."""
    adv = settings_dict.get("advanced", {})
    return {"template": adv.get("naming_template", DEFAULT_TEX_TEMPLATE),
            "material": adv.get("material_template", DEFAULT_MAT_TEMPLATE),
            "style": str(adv.get("naming_style", "VERBOSE")).lower()}


def apply_machine_defaults(scene):
    """Seed a scene's settings from the add-on preferences (plan §8).

    Runs once per scene (`initialized`), from the load_post handler and from
    register(). Nothing is written when there is nothing to seed, so opening
    a file does not mark it modified on a default setup.
    """
    from . import prefs
    p = prefs.get_prefs()
    if p is None:
        return False
    fb = getattr(scene, "fastbake", None)
    if fb is None or fb.initialized:
        return False
    root = (getattr(p, "output_root", "") or "").strip()
    review = bool(getattr(p, "review_default", False))
    if not root and not review:
        return False  # nothing to seed — leave the scene (and the file) alone
    if root and fb.advanced.output_dir in ("", "//textures/"):
        fb.advanced.output_dir = root
    if review:
        fb.review_stages = True
    fb.initialized = True
    return True


classes = (FastBakeAdvanced, FastBakeSettings)


def register():
    bpy.types.Scene.fastbake = PointerProperty(type=FastBakeSettings)


def unregister():
    del bpy.types.Scene.fastbake
