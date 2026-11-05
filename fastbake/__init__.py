# FastBake — one click from high-poly to a baked, game-ready quad asset.
# GPL-3.0-or-later. See blender_manifest.toml.

bl_info = {
    "name": "FastBake",
    "author": "Alvis Misjuns",
    "version": (0, 2, 1),
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


# --- handlers ----------------------------------------------------------------
# The pipeline state is module-level (it must survive redraws), so reading a
# new file has to reset it explicitly: Blender terminates the modal driver
# without ever calling modal() again, and a stale STATE["running"] would lock
# the panel and keep the HUD on screen for the rest of the session.

def _load_pre(*_args):
    from .pipeline import driver
    driver.shutdown()          # cancel the stage, roll back, reset state


def _load_post(*_args):
    import bpy
    from . import props
    from .pipeline import driver, state
    from .ui import hud
    # belt and braces: whatever the modal operator did on its way out, the
    # newly loaded file starts idle
    driver._remove_timer()
    hud.disable()
    state.reset()
    for scene in bpy.data.scenes:
        props.apply_machine_defaults(scene)


def _register_handlers():
    import bpy
    from bpy.app.handlers import persistent
    # persistent() marks the function in place (it returns the same object),
    # without it Blender drops the handler on the first file load — exactly
    # the load we need it for.
    for fn in (_load_pre, _load_post):
        if not hasattr(fn, "_bpy_persistent"):
            persistent(fn)
    _unregister_handlers()          # never register twice (Reload Scripts)
    bpy.app.handlers.load_pre.append(_load_pre)
    bpy.app.handlers.load_post.append(_load_post)


def _unregister_handlers():
    import bpy
    for handlers, fn in ((bpy.app.handlers.load_pre, _load_pre),
                         (bpy.app.handlers.load_post, _load_post)):
        while fn in handlers:
            handlers.remove(fn)


def _seed_machine_defaults():
    """Seed every open scene from the add-on preferences.

    While an extension is being installed or enabled, bpy.data is a
    _RestrictData proxy and any datablock access raises — that is exactly
    when register() runs from Install from Disk. Defer to the first timer
    tick in that case; a plain Enable/Reload Scripts seeds immediately."""
    import bpy
    from . import props
    try:
        scenes = list(bpy.data.scenes)
    except AttributeError:
        bpy.app.timers.register(_seed_machine_defaults, first_interval=0.0)
        return None
    for scene in scenes:
        props.apply_machine_defaults(scene)
    return None


def register():
    import bpy
    from . import presets, props
    for mod in _modules():
        for cls in getattr(mod, "classes", ()):
            bpy.utils.register_class(cls)
    props.register()
    _register_handlers()
    _seed_machine_defaults()
    try:
        presets.ensure_factory_presets()
    except OSError:
        pass


def unregister():
    import bpy
    from . import props
    from .pipeline import driver
    from .ui import hud
    # stop a running pipeline (cancel stage/subprocess, remove the modal
    # timer, roll back) before its operator classes are unregistered
    driver.shutdown()
    hud.disable()
    _unregister_handlers()
    props.unregister()
    for mod in reversed(_modules()):
        for cls in reversed(getattr(mod, "classes", ())):
            bpy.utils.unregister_class(cls)
