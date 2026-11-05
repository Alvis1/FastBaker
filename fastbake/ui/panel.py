# The FastBake N-panel. Three states: idle / running / stage gate.

import time

import bpy
from bpy.props import StringProperty

from ..pipeline.state import STATE

# Which remesher Auto will pick, re-checked every few seconds rather than on
# every redraw: the external-binary lookup globs the file system.
_ENGINE_CACHE = {"at": -1e9, "value": None}
_ENGINE_TTL = 5.0


def _engine_status():
    now = time.monotonic()
    if now - _ENGINE_CACHE["at"] > _ENGINE_TTL:
        from ..engine import autoremesher, core
        if core.core_available():
            value = "core"
        elif autoremesher.find_binary():
            value = "binary"
        else:
            value = None
        _ENGINE_CACHE.update(at=now, value=value)
    return _ENGINE_CACHE["value"]


def _draw_engine_status(layout, engine):
    """One line under the Engine setting saying what will actually run."""
    if engine == "QUADRIFLOW":
        return
    status = _engine_status()
    if status == "core":
        layout.label(text="AutoRemesher engine found", icon="CHECKMARK")
    elif status == "binary":
        layout.label(text="AutoRemesher: external program", icon="CHECKMARK")
    else:
        col = layout.column(align=True)
        col.label(text="AutoRemesher not found: QuadriFlow will run", icon="ERROR")
        col.label(text="Install the AutoRemesher extension to use it")


def _info_popup(context, text, title="FastBake", icon="INFO"):
    def draw(menu, _context):
        for line in str(text).split("\n"):
            menu.layout.label(text=line)

    context.window_manager.popup_menu(draw, title=title, icon=icon)


class FASTBAKE_OT_bake(bpy.types.Operator):
    bl_idname = "fastbake.bake"
    bl_label = "Bake"
    bl_description = ("Merge, retopologize, unwrap and bake the selected "
                      "mesh(es) into a game-ready asset")

    @classmethod
    def poll(cls, context):
        return not STATE["running"]

    def execute(self, context):
        from ..pipeline import driver, preflight
        ctx, err = preflight.make_context(context)
        if err:
            _info_popup(context, err)
            return {"CANCELLED"}
        # Unsaved .blend + the //relative default: ask for a folder instead of
        # writing the textures somewhere the user will never find (plan §5 St0)
        if ctx.flags.get("needs_output_dir") and not bpy.app.background:
            bpy.ops.fastbake.set_output_dir("INVOKE_DEFAULT")
            return {"CANCELLED"}
        # Re-bake guard: preflight.make_context flags an unchanged, already
        # processed selection; ask before doing redundant work.
        if ctx.flags.get("already_baked") and not bpy.app.background:
            bpy.ops.fastbake.confirm_rebake("INVOKE_DEFAULT")
            return {"CANCELLED"}
        if not driver.launch(context, driver.build_bake_stages(), ctx):
            self.report({"ERROR"}, "FastBake could not start — see the console")
            return {"CANCELLED"}
        return {"FINISHED"}


class FASTBAKE_OT_set_output_dir(bpy.types.Operator):
    bl_idname = "fastbake.set_output_dir"
    bl_label = "Where should the textures go?"
    bl_description = "Choose the folder baked textures are written to"
    bl_options = {"INTERNAL"}

    directory: StringProperty(
        name="Output folder", subtype="DIR_PATH", default="",
        description="Folder the baked textures are saved into")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        col = self.layout.column()
        col.label(text="This .blend has never been saved, so the default",
                  icon="ERROR")
        col.label(text="//textures/ has no location on disk.")
        col.separator()
        col.prop(self, "directory")

    def execute(self, context):
        path = (self.directory or "").strip()
        if not path:
            self.report({"ERROR"}, "FastBake needs an output folder")
            return {"CANCELLED"}
        context.scene.fastbake.advanced.output_dir = path
        # the setting is stored either way; run the bake with it now
        return bpy.ops.fastbake.bake("INVOKE_DEFAULT")


class FASTBAKE_OT_confirm_rebake(bpy.types.Operator):
    bl_idname = "fastbake.confirm_rebake"
    bl_label = "Already baked"
    bl_description = "This selection was already processed"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        from ..pipeline import preflight
        ctx, err = preflight.make_context(context)
        if err:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}
        self._msg = ctx.flags.get("already_baked", "Already baked.")
        return context.window_manager.invoke_props_dialog(self, confirm_text="Bake Again")

    def draw(self, context):
        col = self.layout.column()
        col.label(text=getattr(self, "_msg", "Already baked."), icon="INFO")
        col.label(text="Bake again anyway?")

    def execute(self, context):
        from ..pipeline import driver, preflight
        ctx, err = preflight.make_context(context)
        if err:
            return {"CANCELLED"}
        ctx.flags["force"] = True
        if not driver.launch(context, driver.build_bake_stages(), ctx):
            self.report({"ERROR"}, "FastBake could not start — see the console")
            return {"CANCELLED"}
        return {"FINISHED"}


class FASTBAKE_PT_main(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FastBake"
    bl_label = ""
    bl_options = {"HIDE_HEADER"}

    def draw(self, context):
        layout = self.layout
        # large header
        header = layout.row(align=True)
        header.scale_y = 1.6
        header.alignment = "CENTER"
        header.label(text="F A S T B A K E")

        if STATE["running"]:
            self.draw_running(layout)
        else:
            self.draw_idle(context, layout)

    # -- states ---------------------------------------------------------------

    def draw_error(self, layout):
        """Last failure stays visible until the next run starts — the popup
        is gone the moment the mouse moves."""
        box = layout.box()
        col = box.column(align=True)
        col.label(text="Last run failed:", icon="ERROR")
        for line in str(STATE["error"]).split("\n"):
            col.label(text=line)
        if STATE.get("log_text"):
            col.operator("fastbake.copy_log", icon="COPYDOWN")

    def draw_idle(self, context, layout):
        fb = context.scene.fastbake

        if STATE.get("error"):
            self.draw_error(layout)

        col = layout.column()
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(fb, "quads")
        col.prop(fb, "tex_size")
        col.prop(fb, "roughness_to_alpha")
        col.prop(fb, "review_stages")

        # BAKE closes the settings group — no separator above it.
        bake_row = layout.row()
        bake_row.scale_y = 2.4
        bake_row.operator("fastbake.bake", text="B A K E")

        layout.separator()
        lm = layout.row()
        lm.scale_y = 1.2
        lm.operator("fastbake.lightmap", text="Lightmap", icon="LIGHT_SUN")

        layout.separator()
        row = layout.row(align=True)
        row.menu("FASTBAKE_MT_presets", text="Presets", icon="PRESET")
        row.operator("fastbake.preset_save", text="", icon="ADD")
        row.operator("fastbake.preset_delete", text="", icon="REMOVE")

        box = layout.box()
        toggle = box.row()
        toggle.prop(fb, "show_advanced", text="Advanced", emboss=False,
                    icon="TRIA_DOWN" if fb.show_advanced else "TRIA_RIGHT")
        if fb.show_advanced:
            adv = fb.advanced
            col = box.column()
            col.use_property_split = True
            col.use_property_decorate = False
            col.prop(adv, "output_dir")
            col.prop(adv, "engine")
            _draw_engine_status(col, adv.engine)
            col.prop(adv, "merge_detail")
            col.prop(adv, "merge_mode")
            if adv.merge_mode == "OUTER_SHELL":
                col.prop(adv, "shell_probe")
                col.prop(adv, "shell_max_bridged")
            col.prop(adv, "model_type")
            col.prop(adv, "sharp_edge")
            col.prop(adv, "protect_view")
            col.separator()
            col.prop(adv, "margin_px")
            col.prop(adv, "ssaa")
            col.prop(adv, "normal_y")
            col.separator()
            col.prop(adv, "naming_style")
            col.prop(adv, "naming_template")
            col.prop(adv, "material_template")

    def draw_running(self, layout):
        if STATE["waiting"]:
            box = layout.box()
            col = box.column()
            col.label(text=STATE["gate_summary"] or "Stage done", icon="CHECKMARK")
            row = layout.row()
            row.scale_y = 1.8
            row.operator("fastbake.continue_stage", text="Continue")
            row = layout.row(align=True)
            row.operator("fastbake.adjust_stage", text="Adjust...")
            row.operator("fastbake.redo_stage", text="Redo")
            row.operator("fastbake.cancel_run", text="Cancel")
            return

        col = layout.column()
        row = col.row()
        row.scale_y = 1.4
        row.label(text=STATE["message"] or "WORKING")
        col.label(text=STATE["detail"])
        bar = col.row()
        bar.scale_y = 1.5
        bar.progress(factor=STATE["overall"],
                     text="%d%%" % int(STATE["overall"] * 100))
        cancel = layout.row()
        cancel.scale_y = 1.3
        cancel.operator("fastbake.cancel_run", text="Cancel")


classes = (FASTBAKE_OT_bake, FASTBAKE_OT_set_output_dir,
           FASTBAKE_OT_confirm_rebake, FASTBAKE_PT_main)
