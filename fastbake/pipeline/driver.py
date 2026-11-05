# Pipeline driver: a modal operator with a window timer in interactive
# Blender, a plain blocking loop in background mode (headless tests).
# Consumes the stage queue in pipeline.state.STATE.

import os
import shutil
import time
import traceback
from datetime import datetime

import bpy

from .. import compat
from . import state
from .context import StageError, purge_stale_tmpdirs
from .state import STATE

LOG_TEXT_NAME = "FastBake.log"


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
    """Entry point used by the BAKE / Lightmap operators.

    Returns True when the pipeline is running (or, in background mode, has
    finished). A False return leaves NO trace: the state, the HUD and the
    status bar are exactly as before the call, so a failed start can never
    wedge the add-on in the running state.
    """
    if STATE["running"]:
        return False
    purge_stale_tmpdirs()  # leftovers of runs that died before teardown
    state.reset()
    STATE.update(running=True, stages=stages, ctx=ctx, error=None,
                 log_text=None, cancel_wait=0, draining=False)
    _set_labels()
    if bpy.app.background:
        _run_blocking()
        return True
    # plan §4 undo discipline: one push at pipeline start (the anchor the
    # make-single-user / apply-transforms step is reverted to), one per stage
    _push_undo("FastBake: start")
    from ..ui import hud
    hud.enable()
    try:
        result = bpy.ops.fastbake.run_pipeline("INVOKE_DEFAULT")
    except Exception:
        ctx.log(traceback.format_exc())
        result = None
    if result != {"RUNNING_MODAL"}:
        # No modal timer exists, so nothing would ever tick _step() again.
        ctx.log("the pipeline operator did not start (%r)" % (result,))
        hud.disable()
        _status_text(None)
        _remove_timer()
        state.reset()
        state.tag_redraw()
        return False
    return True


def _set_labels():
    if STATE.get("draining"):
        # cancel pressed while Cycles owns the bake target: we cannot abort
        # the job, so we wait it out (see _step) and say so
        STATE["message"] = "CANCELLING"
        STATE["detail"] = "Waiting for Cycles to finish…"
        STATE["overall"] = state.compute_overall()
        if not bpy.app.background:
            _status_text("FastBake: waiting for Cycles to finish…")
        return
    stages = STATE["stages"]
    i = STATE["index"]
    if STATE["waiting"] and i > 0:
        # parked at a review gate: the next stage has not started yet
        STATE["message"] = "PAUSED"
        STATE["detail"] = "review %s" % stages[i - 1].stage_id
    elif i < len(stages):
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


def _stage_provenance(ctx, stage):
    """Stage exit = undo push + provenance update (plan §4).

    The full manifest is written by the material stage; this is the running
    breadcrumb, so an inspected mid-run object says which stage produced it.
    It is removed from the finished result again by _finish().
    """
    obj = ctx.obj(ctx.result_name) or ctx.obj(ctx.work_name)
    if obj is None:
        return
    try:
        obj["fastbake_progress"] = {
            "stage": str(stage.stage_id),
            "base": str(ctx.base),
            "run": int(ctx.run),
            "date": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception:
        pass  # provenance is never worth failing a run over


# --- shared step logic (modal + blocking) ------------------------------------

def _step():
    """Advance the pipeline one tick.

    Returns 'RUNNING' | 'WAITING' | 'DONE' | 'CANCELLED' | 'ERROR'.
    """
    ctx = STATE["ctx"]
    stages = STATE["stages"]

    if STATE["cancel"]:
        # A running Cycles bake job cannot be aborted from Python, and freeing
        # its target image/object mid-job is a use-after-free (crash). So we
        # wait for the job to drain — with NO time cap: a 8k x 8k map takes
        # minutes, and "roll back anyway" would corrupt exactly the heavy
        # bakes users cancel.
        if compat.bake_job_running():
            STATE["cancel_wait"] = STATE.get("cancel_wait", 0) + 1
            if not STATE.get("draining"):
                STATE["draining"] = True
                ctx.log("cancel requested — waiting for the running Cycles "
                        "bake job to finish before rolling back")
            return "RUNNING"
        STATE["draining"] = False
        if STATE["started"] and STATE["index"] < len(stages):
            try:
                stages[STATE["index"]].cancel(ctx)
            except Exception:
                ctx.log(traceback.format_exc())
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
            _fail(ctx, str(exc), traceback.format_exc())
            return "ERROR"
        except Exception:
            tb = traceback.format_exc()
            ctx.log(tb)
            _fail(ctx, "Unexpected error in %s stage — see console" % stage.stage_id, tb)
            return "ERROR"
        STATE["started"] = True
        STATE["stage_progress"] = 0.0

    try:
        progress = stage.poll(ctx)
    except StageError as exc:
        _fail(ctx, str(exc), traceback.format_exc())
        return "ERROR"
    except Exception:
        tb = traceback.format_exc()
        ctx.log(tb)
        _fail(ctx, "Unexpected error in %s stage — see console" % stage.stage_id, tb)
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
    _stage_provenance(ctx, stage)

    review = ctx.settings.get("review_stages") and stage.gate and not bpy.app.background
    STATE["index"] += 1
    STATE["started"] = False
    # the next stage has not run yet: without this reset compute_overall()
    # counts it as 100% done while parked at a gate (progress overshoot)
    STATE["stage_progress"] = 0.0
    if review:
        STATE["waiting"] = True
        STATE["gate_summary"] = stage.summary(ctx)
        _set_labels()
        return "WAITING"
    if STATE["index"] >= len(stages):
        _finish(ctx)
        return "DONE"
    _set_labels()
    return "RUNNING"


def _rollback(ctx, why):
    ctx.log("rollback: " + why)
    for hook in reversed(ctx.undo_hooks):
        try:
            hook()
        except Exception:
            ctx.log(traceback.format_exc())
    ctx.undo_hooks.clear()
    ctx.rollback_to((0, 0, 0))
    _restore_sources(ctx)
    _teardown(why)


def _restore_sources(ctx):
    for name, (hv, hr) in ctx.src_visibility.items():
        obj = bpy.data.objects.get(name)
        if obj:
            obj.hide_viewport = hv
            obj.hide_render = hr
            try:
                obj.hide_set(False)
            except RuntimeError:
                pass  # not in the current view layer


def _write_log(ctx, message, tb=None):
    """Put the whole run log in a Text datablock so it can be read and copied.

    Returns the text name, or None when no log could be written.
    """
    lines = ["FastBake failed: %s" % message, ""]
    if ctx is not None:
        lines.extend(str(l) for l in ctx.report_lines)
    if tb:
        lines.append("")
        lines.append(tb.rstrip())
    try:
        text = bpy.data.texts.get(LOG_TEXT_NAME)
        if text is None:
            text = bpy.data.texts.new(LOG_TEXT_NAME)
        text.clear()
        text.write("\n".join(lines))
        return text.name
    except Exception:
        return None


def _fail(ctx, message, tb=None):
    STATE["error"] = message
    STATE["log_text"] = _write_log(ctx, message, tb)
    _rollback(ctx, "error: " + message)
    if not bpy.app.background:
        _error_popup(message)


def _finish(ctx):
    # transform-restore backups are only needed for rollback; a successful
    # run no longer can roll back, so free the zero-user mesh copies
    for name in ctx.flags.get("transform_backups", ()):
        me = bpy.data.meshes.get(name)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    ctx.undo_hooks.clear()
    result = ctx.obj(ctx.result_name)
    if result is not None and "fastbake_progress" in result:
        del result["fastbake_progress"]  # the full manifest superseded it
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
    try:
        lines.append("Saved to: " + ctx.output_dir(create=False))
    except StageError:
        pass
    lines += ctx.notices

    def draw(menu, _context):
        for line in lines:
            menu.layout.label(text=line)

    bpy.context.window_manager.popup_menu(draw, title="FastBake", icon="CHECKMARK")


def _teardown(why):
    STATE["running"] = False
    STATE["waiting"] = False
    STATE["draining"] = False
    STATE["message"] = ""
    _remove_tmpdir(STATE["ctx"])
    if not bpy.app.background:
        from ..ui import hud
        hud.disable()
        _status_text(None)
        state.tag_redraw()


def _remove_tmpdir(ctx):
    """Delete the per-run temp dir (OBJ dumps, denoise EXR, logs).

    Called only on final teardown (done/cancelled/error), never between
    stages — the lightmap stage keeps temp EXRs there mid-run. Contexts that
    never needed scratch space never created one."""
    if ctx is None:
        return
    path = ctx.tmpdir_if_created()
    if not path:
        return
    tmp = os.path.realpath(path)
    for saved in ctx.files.values():
        if os.path.realpath(saved).startswith(tmp + os.sep):
            return  # should not happen (output_dir never points here) — be safe
    shutil.rmtree(path, ignore_errors=True)


def _error_popup(message):
    log_name = STATE.get("log_text")

    def draw(menu, _context):
        col = menu.layout.column()
        for line in str(message).split("\n"):
            col.label(text=line)
        if log_name:
            col.separator()
            col.label(text="Full log: Text datablock '%s'" % log_name)
            col.operator("fastbake.copy_log", icon="COPYDOWN")

    bpy.context.window_manager.popup_menu(draw, title="FastBake failed", icon="ERROR")


def _remove_timer():
    timer = STATE.pop("timer", None)
    if timer is None:
        return
    try:
        bpy.context.window_manager.event_timer_remove(timer)
    except Exception:
        pass


def _abort_external(why):
    """Blender terminated the modal operator from the outside (file load,
    new file, Reload Scripts). No datablock work here — the file may be
    mid-swap and load_pre has already rolled the run back; this only makes
    sure the add-on is not left wedged in the running state."""
    if STATE["running"]:
        ctx = STATE["ctx"]
        stages = STATE["stages"]
        if ctx is not None and STATE["started"] and STATE["index"] < len(stages):
            try:
                stages[STATE["index"]].cancel(ctx)   # kills subprocesses
            except Exception:
                traceback.print_exc()
        _remove_tmpdir(ctx)          # files only: no datablock work here
        print("[FastBake] pipeline aborted: %s" % why)
    _remove_timer()
    if not bpy.app.background:
        from ..ui import hud
        hud.disable()
        _status_text(None)
    state.reset()
    if not bpy.app.background:
        state.tag_redraw()


def shutdown():
    """Stop a running pipeline. Called by the add-on's unregister() and by
    the load_pre handler, so disabling/updating FastBake or opening another
    file mid-run cancels the current stage (which terminates any AutoRemesher
    subprocess), rolls back and resets module state — otherwise the window
    timer leaks and STATE['running'] stays True."""
    if STATE["running"]:
        ctx = STATE["ctx"]
        stages = STATE["stages"]
        if ctx is not None and STATE["started"] and STATE["index"] < len(stages):
            try:
                stages[STATE["index"]].cancel(ctx)
            except Exception:
                traceback.print_exc()
        if ctx is not None and compat.bake_job_running():
            # Decision 3 applies here too: never free the images/objects a
            # live Cycles bake job is still writing into. Tear the driver
            # down and leave the datablocks to the file load / Blender.
            ctx.log("shutdown: Cycles bake job still running — rollback skipped")
            _teardown("add-on disabled")
        elif ctx is not None:
            _rollback(ctx, "add-on disabled")
        else:
            _teardown("add-on disabled")
    _remove_timer()
    state.reset()


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
        # module-level reference so shutdown() (add-on disable) can remove
        # the timer without the operator instance
        STATE["timer"] = self._timer
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            STATE["cancel"] = True
        # Swallow undo/redo while the pipeline owns the scene: ed.undo would
        # delete run datablocks mid-stage or at a review gate. macOS undo is
        # Cmd+Z (oskey), Ctrl covers Ctrl+Z / Ctrl+Shift+Z elsewhere.
        if event.type == "Z" and (event.ctrl or event.oskey):
            return {"RUNNING_MODAL"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            result = _step()
        except Exception:
            # _step guards every stage call, so this is a bug in the driver
            # itself — still, never leave the add-on running with a dead tick
            traceback.print_exc()
            shutdown()
            return {"CANCELLED"}
        state.tag_redraw()
        if result in ("RUNNING", "WAITING"):
            _set_labels()
            return {"PASS_THROUGH"}

        # DONE / CANCELLED / ERROR
        _remove_timer()
        if result == "CANCELLED":
            self.report({"WARNING"}, "FastBake cancelled")
            return {"CANCELLED"}
        if result == "ERROR":
            return {"CANCELLED"}
        return {"FINISHED"}

    def cancel(self, context):
        # Called when Blender ends this modal operator without modal()
        # returning FINISHED/CANCELLED — reading a new file is the common
        # case. Without this the add-on stays "running" forever.
        _abort_external("modal operator cancelled by Blender")


class FASTBAKE_OT_copy_log(bpy.types.Operator):
    bl_idname = "fastbake.copy_log"
    bl_label = "Copy Log"
    bl_description = "Copy the FastBake run log to the clipboard"

    def execute(self, context):
        text = bpy.data.texts.get(LOG_TEXT_NAME)
        if text is None:
            self.report({"WARNING"}, "No FastBake log available")
            return {"CANCELLED"}
        context.window_manager.clipboard = text.as_string()
        self.report({"INFO"}, "FastBake log copied to the clipboard")
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
        # Refresh the snapshot so redo picks up the adjusted values — through
        # the same machine-defaults pass make_context used, otherwise the
        # resolved output root (add-on preferences) silently reverts to the
        # scene's default for the rest of the run.
        from ..props import settings_to_dict
        from . import preflight
        fresh = settings_to_dict(context.scene.fastbake)
        STATE["ctx"].settings = preflight._apply_machine_defaults(fresh) or fresh
        return {"FINISHED"}


classes = (FASTBAKE_OT_run_pipeline, FASTBAKE_OT_copy_log,
           FASTBAKE_OT_continue_stage, FASTBAKE_OT_redo_stage,
           FASTBAKE_OT_cancel_run, FASTBAKE_OT_adjust_stage)
