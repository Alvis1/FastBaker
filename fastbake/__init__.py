# FastBake — one click from high-poly to a baked, game-ready quad asset.
# GPL-3.0-or-later. See blender_manifest.toml.

bl_info = {
    "name": "FastBake",
    "author": "Alvis Misjuns",
    "version": (0, 1, 0),
    "blender": (4, 5, 0),
    "location": "3D Viewport > Sidebar > FastBake",
    "description": "Merge, quad-retopo (AutoRemesher), auto-UV and bake in one click",
    "category": "Baking",
}


def _modules():
    from . import prefs, presets, props
    from .pipeline import driver, lightmap
    from .ui import panel
    return (props, prefs, presets, driver, lightmap, panel)


def register():
    import bpy
    from . import presets, props
    for mod in _modules():
        for cls in getattr(mod, "classes", ()):
            bpy.utils.register_class(cls)
    props.register()
    try:
        presets.ensure_factory_presets()
    except OSError:
        pass


def unregister():
    import bpy
    from . import props
    from .ui import hud
    hud.disable()
    props.unregister()
    for mod in reversed(_modules()):
        for cls in reversed(getattr(mod, "classes", ())):
            bpy.utils.unregister_class(cls)
