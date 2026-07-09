# The FastBake N-panel. Three states: idle / running / stage gate.

import bpy

from ..pipeline.state import STATE


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
            def draw(menu, _context):
                menu.layout.label(text=err)
            context.window_manager.popup_menu(draw, title="FastBake", icon="INFO")
            return {"CANCELLED"}
        # Re-bake guard: preflight.make_context flags an unchanged, already
        # processed selection; ask before doing redundant work.
        if ctx.flags.get("already_baked") and not bpy.app.background:
            bpy.ops.fastbake.confirm_rebake("INVOKE_DEFAULT")
            return {"CANCELLED"}
        driver.launch(context, driver.build_bake_stages(), ctx)
        return {"FINISHED"}


class FASTBAKE_OT_confirm_rebake(bpy.types.Operator):
    bl_idname = "fastbake.confirm_rebake"
    bl_label = "Already baked"
    bl_description = "This selection was already processed"

    def invoke(self, context, event):
        from ..pipeline import preflight
        ctx, _err = preflight.make_context(context)
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
        driver.launch(context, driver.build_bake_stages(), ctx)
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

    def draw_idle(self, context, layout):
        fb = context.scene.fastbake

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
            col.prop(adv, "merge_detail")
            col.prop(adv, "model_type")
            col.prop(adv, "sharp_edge")
            col.prop(adv, "protect_view")
            col.separator()
            col.prop(adv, "margin_px")
            col.prop(adv, "ssaa")
            col.prop(adv, "normal_y")
            col.prop(adv, "naming_style")

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


classes = (FASTBAKE_OT_bake, FASTBAKE_OT_confirm_rebake, FASTBAKE_PT_main)
