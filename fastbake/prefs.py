# Machine-level preferences.

import bpy
from bpy.props import BoolProperty, StringProperty


class FastBakePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    autoremesher_path: StringProperty(
        name="AutoRemesher executable", subtype="FILE_PATH", default="",
        description="Path to the AutoRemesher binary (leave empty to auto-detect)")
    hud_enabled: BoolProperty(
        name="Viewport progress HUD", default=True,
        description="Draw large progress text in the 3D viewport while running")

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "autoremesher_path")
        layout.prop(self, "hud_enabled")


def get_prefs():
    try:
        return bpy.context.preferences.addons[__package__].preferences
    except KeyError:
        return None  # running outside a registered add-on (tests)


classes = (FastBakePreferences,)
