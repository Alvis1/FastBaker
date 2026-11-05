# Machine-level preferences (plan §8: output root, HUD toggle, review default).
# Scene-level decisions live in props.FastBakeSettings and save with the
# .blend; the values here follow the machine and seed new scenes through
# props.apply_machine_defaults().

import bpy

from . import compat
from bpy.props import BoolProperty, StringProperty


class FastBakePreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    autoremesher_path: StringProperty(
        name="AutoRemesher executable", subtype="FILE_PATH", default="",
        description="Path to the AutoRemesher binary (leave empty to auto-detect)")
    output_root: StringProperty(
        name="Output root", subtype="DIR_PATH", default="",
        options=compat.path_prop_options(),
        description="Folder new scenes bake into instead of //textures/ "
                    "(leave empty to keep the blend-relative default)")
    review_default: BoolProperty(
        name="Review each stage by default", default=False,
        description="New scenes start with the per-stage review gates enabled")
    hud_enabled: BoolProperty(
        name="Viewport progress HUD", default=True,
        description="Draw large progress text in the 3D viewport while running")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "autoremesher_path")
        col.separator()
        col.prop(self, "output_root")
        col.prop(self, "review_default")
        col.prop(self, "hud_enabled")


def get_prefs():
    try:
        return bpy.context.preferences.addons[__package__].preferences
    except KeyError:
        return None  # running outside a registered add-on (tests)


classes = (FastBakePreferences,)
