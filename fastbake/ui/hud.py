# Viewport progress HUD. Big blf text drawn top-center over every 3D view
# while the pipeline runs. The driver calls enable()/disable(); never active
# in background mode (blf needs a GPU context).

import blf
import bpy

from .. import prefs
from ..pipeline.state import STATE

_handle = None

_COLOR = (1.0, 0.6, 0.1, 0.9)
_SIZE_PERCENT = 64
_SIZE_MESSAGE = 28
_SIZE_DETAIL = 16


def enable():
    global _handle
    if bpy.app.background or _handle is not None:
        return
    p = prefs.get_prefs()
    if p is not None and not p.hud_enabled:
        return
    _handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw, (), "WINDOW", "POST_PIXEL")


def disable():
    global _handle
    if _handle is None:
        return
    try:
        bpy.types.SpaceView3D.draw_handler_remove(_handle, "WINDOW")
    except ValueError:
        pass  # already removed (e.g. by a reload)
    _handle = None


def _draw():
    if not STATE["running"]:
        return
    region = bpy.context.region
    if region is None:
        return
    ui_scale = bpy.context.preferences.system.ui_scale

    font_id = 0
    blf.enable(font_id, blf.SHADOW)
    blf.shadow(font_id, 3, 0.0, 0.0, 0.0, 0.7)
    blf.shadow_offset(font_id, 2, -2)
    blf.color(font_id, *_COLOR)

    lines = (("%d%%" % round(STATE["overall"] * 100), _SIZE_PERCENT),
             (STATE["message"], _SIZE_MESSAGE),
             (STATE["detail"], _SIZE_DETAIL))
    y = region.height - 40.0 * ui_scale
    for text, size in lines:
        if not text:
            continue
        blf.size(font_id, size * ui_scale)
        width, height = blf.dimensions(font_id, text)
        y -= height
        blf.position(font_id, (region.width - width) * 0.5, y, 0.0)
        blf.draw(font_id, text)
        y -= 10.0 * ui_scale

    blf.disable(font_id, blf.SHADOW)


classes = ()
