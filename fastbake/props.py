# Settings model. Scene-level (saves with the .blend).

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)

TEX_SIZES = [(s, s, "") for s in ("512", "1024", "2048", "4096", "8192")]
LM_SIZES = [(s, s, "") for s in ("256", "512", "1024", "2048", "4096")]


class FastBakeAdvanced(bpy.types.PropertyGroup):
    output_dir: StringProperty(
        name="Output", subtype="DIR_PATH", default="//textures/",
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
               ("AUTOREMESHER", "AutoRemesher", "Requires the AutoRemesher binary or wheel"),
               ("QUADRIFLOW", "QuadriFlow", "Blender built-in")])
    merge_detail: EnumProperty(
        name="Merge detail", default="MED",
        items=[("LOW", "Low", "Voxel = diagonal/150"),
               ("MED", "Medium", "Voxel = diagonal/300"),
               ("HIGH", "High", "Voxel = diagonal/600")])
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


# --- snapshot ---------------------------------------------------------------

_SNAP_KEYS = ("quads", "tex_size", "roughness_to_alpha", "review_stages")
_SNAP_ADV = ("output_dir", "margin_px", "ssaa", "normal_y", "engine",
             "merge_detail", "model_type", "sharp_edge", "protect_view",
             "naming_style", "lm_size", "lm_samples", "lm_gamma", "lm_denoise")


def settings_to_dict(settings):
    d = {k: getattr(settings, k) for k in _SNAP_KEYS}
    d["advanced"] = {k: getattr(settings.advanced, k) for k in _SNAP_ADV}
    return d


def dict_to_settings(d, settings):
    for k in _SNAP_KEYS:
        if k in d:
            try:
                setattr(settings, k, d[k])
            except (TypeError, ValueError):
                pass  # tolerate presets from other versions
    adv = d.get("advanced", {})
    for k in _SNAP_ADV:
        if k in adv:
            try:
                setattr(settings.advanced, k, adv[k])
            except (TypeError, ValueError):
                pass  # tolerate presets from other versions


classes = (FastBakeAdvanced, FastBakeSettings)


def register():
    bpy.types.Scene.fastbake = PointerProperty(type=FastBakeSettings)


def unregister():
    del bpy.types.Scene.fastbake
