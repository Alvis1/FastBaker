# Pipeline driver: a modal operator with a window timer in interactive
# Blender, a plain blocking loop in background mode (headless tests).
# Consumes the stage queue in pipeline.state.STATE.

import time
import traceback

import bpy

from . import state
from .context import StageError
from .state import STATE


def build_bake_stages():
    from . import bake, material, merge, pack, preflight, retopo, uvs
    return [preflight.PreflightStage(),
            preflight.ApplyTransformsStage(),
            merge.MergeStage(),
            retopo.RetopoStage(),
            uvs.UVStage(),
            bake.BakeStage(),
            pack.PackStage(),
            material.MaterialStage()]


def build_lightmap_stages():
    from . import lightmap, preflight
    return [preflight.LightmapPreflightStage(),
            lightmap.LightmapStage()]


def launch(context, stages, ctx):
    """Entry point used by the BAKE / Lightmap operators."""
    if STATE["running"]:
        return False
    state.reset()
    STATE.update(running=True, stages=stages, ctx=ctx, error=None)
    _set_labels()
    if bpy.app.background:
        _run_blocking()
        return True
    from ..ui import hud
    hud.enable()
    return bpy.ops.fastbake.run_pipeline("INVOKE_DEFAULT") == {"RUNNING_MODAL"}


def _set_labels():
    stages = STATE["stages"]
    i = STATE["index"]
    if i < len(stages):
        STATE["message"] = stages[i].label.upper()
        STATE["detail"] = "stage %d/%d" % (i + 1, len(stages))
    STATE["overall"] = state.compute_overall()
    if not bpy.app.background:
        _status_text("FastBake: %s (%s) %d%%" % (
            STATE["message"].capitalize(), STATE["detail"], STATE["overall"] * 100))


def _status_text(text):
    try:
        bpy.context.workspace.status_text_set(text)
    except Exception:
        pass


def _push_undo(label):
    if bpy.app.background:
        return
    try:
        bpy.ops.ed.undo_push(message=label)
    except Exception:
        pass


# --- shared step logic (modal + blocking) ------------------------------------

def _step():
    """Advance the pipeline one tick.

    Returns 'RUNNING' | 'WAITING' | 'DONE' | 'CANCELLED' | 'ERROR'.
    """
    ctx = STATE["ctx"]
    stages = STATE["stages"]

    if STATE["cancel"]:
        if STATE["started"] and STATE["index"] < len(stages):
            try:
                stages[STATE["index"]].cancel(ctx)
            except Exception:
                pass
        _rollback(ctx, "Cancelled")
        return "CANCELLED"

    if STATE["waiting"]:
        if STATE["redo"]:
            STATE["redo"] = False
            STATE["waiting"] = False
            i = STATE["index"]
            counts = getattr(stages[i], "_snapshot", None)
            if counts is not None:
                ctx.rollback_to(counts)
            STATE["started"] = False
            _set_labels()
        return "WAITING"

    if STATE["index"] >= len(stages):
        _finish(ctx)
        return "DONE"

    stage = stages[STATE["index"]]

    if not STATE["started"]:
        stage._snapshot = ctx.snapshot_counts()
        ctx.log("stage start: " + stage.stage_id)
        try:
            stage.start(ctx)
        except StageError as exc:
            _fail(ctx, str(exc))
            return "ERROR"
        except Exception:
            ctx.log(traceback.format_exc())
            _fail(ctx, "Unexpected error in %s stage — see console" % stage.stage_id)
            return "ERROR"
        STATE["started"] = True
        STATE["stage_progress"] = 0.0

    try:
        progress = stage.poll(ctx)
    except StageError as exc:
        _fail(ctx, str(exc))
        return "ERROR"
    except Exception:
        ctx.log(traceback.format_exc())
        _fail(ctx, "Unexpected error in %s stage — see console" % stage.stage_id)
        return "ERROR"

    if progress is not None:
        STATE["stage_progress"] = max(0.0, min(1.0, float(progress)))
        STATE["overall"] = state.compute_overall()
        return "RUNNING"

    # stage finished
    STATE["stage_progress"] = 1.0
    STATE["overall"] = state.compute_overall()
    ctx.log("stage done: " + stage.stage_id)
    _push_undo("FastBake: " + stage.stage_id)

    review = ctx.settings.get("review_stages") and stage.gate and not bpy.app.background
    STATE["index"] += 1
    STATE["started"] = False
    if STATE["index"] < len(stages):
        _set_labels()
    if review:
        STATE["waiting"] = True
        STATE["gate_summary"] = stage.summary(ctx)
        return "WAITING"
    if STATE["index"] >= len(stages):
        _finish(ctx)
        return "DONE"
    return "RUNNING"


def _rollback(ctx, why):
    ctx.log("rollback: " + why)
    ctx.rollback_to((0, 0, 0))
    _restore_sources(ctx)
    _teardown(why)


def _restore_sources(ctx):
    for name, (hv, hr) in ctx.src_visibility.items():
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_viewport = hv
            obj.hide_render = hr


def _fail(ctx, message):
    STATE["error"] = message
    _rollback(ctx, "error: " + message)
    if not bpy.app.background:
        _error_popup(message)


def _finish(ctx):
    _teardown("done")
    if bpy.app.background:
        return
    lines = ["FastBake finished:"]
    if ctx.result_name:
        lines.append("Mesh: " + ctx.result_name)
    for key, name in ctx.images.items():
        lines.append("%s: %s" % (key.capitalize(), name))
    if ctx.material_name:
        lines.append("Material: " + ctx.material_name)
    lines += ctx.notices

    def draw(menu, _context):
        for line in lines:
            menu.layout.label(text=line)

    bpy.context.window_manager.popup_menu(draw, title="FastBake", icon="CHECKMARK")


def _teardown(why):
    STATE["running"] = False
    STATE["waiting"] = False
    STATE["message"] = ""
    if not bpy.app.background:
        from ..ui import hud
        hud.disable()
        _status_text(None)
        state.tag_redraw()


def _error_popup(message):
    def draw(menu, _context):
        menu.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title="FastBake failed", icon="ERROR")


def _run_blocking():
    """Headless mode: run the whole queue synchronously."""
    while True:
        result = _step()
        if result == "RUNNING":
            time.sleep(0.02)
            continue
        if result == "WAITING":  # gates are interactive-only; skip
            STATE["waiting"] = False
            continue
        break
    if STATE["error"]:
        raise RuntimeError("FastBake pipeline failed: %s" % STATE["error"])


# --- operators ----------------------------------------------------------------

class FASTBAKE_OT_run_pipeline(bpy.types.Operator):
    bl_idname = "fastbake.run_pipeline"
    bl_label = "FastBake Pipeline"
    bl_options = {"INTERNAL"}

    _timer = None

    def invoke(self, context, event):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            STATE["cancel"] = True
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        result = _step()
        state.tag_redraw()
        if result in ("RUNNING", "WAITING"):
            _set_labels()
            return {"PASS_THROUGH"}

        # DONE / CANCELLED / ERROR
        context.window_manager.event_timer_remove(self._timer)
        if result == "CANCELLED":
            self.report({"WARNING"}, "FastBake cancelled")
            return {"CANCELLED"}
        if result == "ERROR":
            return {"CANCELLED"}
        return {"FINISHED"}


class FASTBAKE_OT_continue_stage(bpy.types.Operator):
    bl_idname = "fastbake.continue_stage"
    bl_label = "Continue"
    bl_description = "Continue with the next FastBake stage"

    @classmethod
    def poll(cls, context):
        return STATE["running"] and STATE["waiting"]

    def execute(self, context):
        STATE["waiting"] = False
        return {"FINISHED"}


class FASTBAKE_OT_redo_stage(bpy.types.Operator):
    bl_idname = "fastbake.redo_stage"
    bl_label = "Redo Stage"
    bl_description = "Discard this stage's result and run it again (after adjusting)"

    @classmethod
    def poll(cls, context):
        return STATE["running"] and STATE["waiting"] and STATE["index"] > 0

    def execute(self, context):
        # step back to the stage that just finished
        STATE["index"] = max(0, STATE["index"] - 1)
        STATE["redo"] = True
        return {"FINISHED"}


class FASTBAKE_OT_cancel_run(bpy.types.Operator):
    bl_idname = "fastbake.cancel_run"
    bl_label = "Cancel"
    bl_description = "Cancel the FastBake run and roll back"

    @classmethod
    def poll(cls, context):
        return STATE["running"]

    def execute(self, context):
        STATE["cancel"] = True
        STATE["waiting"] = False
        return {"FINISHED"}


class FASTBAKE_OT_adjust_stage(bpy.types.Operator):
    bl_idname = "fastbake.adjust_stage"
    bl_label = "Adjust"
    bl_description = "Adjust the finished stage's settings, then redo or continue"

    @classmethod
    def poll(cls, context):
        if not (STATE["running"] and STATE["waiting"] and STATE["index"] > 0):
            return False
        stage = STATE["stages"][STATE["index"] - 1]
        return bool(stage.adjust_props)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        stage = STATE["stages"][STATE["index"] - 1]
        adv = context.scene.fastbake.advanced
        col = self.layout.column()
        for prop in stage.adjust_props:
            col.prop(adv, prop)

    def execute(self, context):
        # refresh the snapshot so redo picks up the adjusted values
        from ..props import settings_to_dict
        STATE["ctx"].settings = settings_to_dict(context.scene.fastbake)
        return {"FINISHED"}


classes = (FASTBAKE_OT_run_pipeline, FASTBAKE_OT_continue_stage,
           FASTBAKE_OT_redo_stage, FASTBAKE_OT_cancel_run,
           FASTBAKE_OT_adjust_stage)
