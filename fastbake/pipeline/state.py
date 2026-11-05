# Module-level pipeline state. Lives outside any operator instance so it
# survives redraws and is inspectable by the panel. Only one pipeline can run
# at a time.
#
# NOTE: this module also survives a file load, which is why the driver
# installs a load_pre handler and a modal cancel() callback — a stale
# "running" here would lock the panel for the rest of the session.

STATE = {
    "running": False,
    "stages": [],          # list of Stage instances
    "index": 0,            # current stage index
    "started": False,      # current stage started?
    "ctx": None,           # PipelineContext
    "stage_progress": 0.0, # 0..1 within current stage
    "overall": 0.0,        # 0..1 across the run
    "message": "",         # big label ("RETOPOLOGIZING")
    "detail": "",          # small label ("stage 3/7")
    "waiting": False,      # parked at a review gate
    "gate_summary": "",    # summary text shown at the gate
    "cancel": False,       # cancel requested
    "cancel_wait": 0,      # ticks spent waiting for a Cycles job to drain
    "draining": False,     # cancelled, waiting out a running bake job
    "redo": False,         # redo of current stage requested
    "error": None,         # user-facing error string after failure
    "log_text": None,      # name of the Text datablock holding the run log
}

# Relative durations for the overall progress bar.
STAGE_WEIGHTS = {
    "preflight": 2, "transforms": 1, "merge": 10, "retopo": 40,
    "uv": 12, "bake": 25, "pack": 4, "material": 2, "lightmap": 30,
}


def reset():
    """Back to idle. `error`/`log_text` survive on purpose — the panel shows
    the last failure until the next run starts (launch clears them)."""
    STATE.update(running=False, stages=[], index=0, started=False, ctx=None,
                 stage_progress=0.0, overall=0.0, message="", detail="",
                 waiting=False, gate_summary="", cancel=False, cancel_wait=0,
                 draining=False, redo=False)


def compute_overall():
    stages = STATE["stages"]
    if not stages:
        return 0.0
    weights = [STAGE_WEIGHTS.get(s.stage_id, 5) for s in stages]
    total = sum(weights)
    done = sum(weights[:STATE["index"]])
    cur = weights[STATE["index"]] * STATE["stage_progress"] if STATE["index"] < len(stages) else 0.0
    return min(1.0, (done + cur) / total)


def tag_redraw():
    import bpy
    if bpy.app.background:
        return
    wm = bpy.context.window_manager
    if not wm:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
