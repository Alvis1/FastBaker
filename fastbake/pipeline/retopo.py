# Retopology stage: AutoRemesher subprocess (preferred) with a blocking
# QuadriFlow fallback. Async — poll() drives the subprocess; QuadriFlow runs
# synchronously (seconds-scale, allowed to block).

import bpy

from .. import naming
from ..engine import autoremesher
from .context import Stage, StageError


class RetopoStage(Stage):
    stage_id = "retopo"
    label = "Retopologizing"
    gate = True
    adjust_props = ("engine", "model_type", "sharp_edge")

    def __init__(self):
        self.job = None
        # Kept so a gate Redo can rebuild the work scaffold after we removed
        # it on success (the mesh datablock is deliberately left alive).
        self._work_mesh_name = None
        self._work_matrix = None

    def start(self, ctx):
        self.job = None
        work = ctx.obj(ctx.work_name)
        if work is None:
            work = self._resurrect_work(ctx)
        if work is None:
            raise StageError("Retopo: work mesh is missing")

        engine = ctx.settings["advanced"]["engine"]
        if engine in ("AUTO", "AUTOREMESHER"):
            exe = autoremesher.find_binary()
            if exe:
                self.job = autoremesher.AutoRemesherJob(exe, ctx)
                try:
                    self.job.start(work, ctx.settings["quads"],
                                   self._sharp_edge(ctx))
                    return
                except OSError as exc:
                    ctx.log("autoremesher launch failed: %s" % exc)
                    self.job = None
            if engine == "AUTOREMESHER":
                ctx.notice("AutoRemesher not available — used QuadriFlow instead")
            else:
                ctx.log("AutoRemesher not found; using QuadriFlow")
        self._quadriflow(ctx)

    def poll(self, ctx):
        if self.job is None:
            return None
        if self.job.running():
            return self.job.progress()

        # subprocess finished
        job, self.job = self.job, None
        job.close()
        if job.succeeded():
            try:
                self._adopt_autoremesher(ctx, job)
                return None
            except (ValueError, OSError) as exc:
                ctx.log("autoremesher output unusable: %s" % exc)
        else:
            code = job.proc.returncode if job.proc else "?"
            ctx.log("autoremesher failed (exit %s): %s"
                    % (code, job.error_excerpt()))
        ctx.notice("AutoRemesher failed — fell back to QuadriFlow")
        self._quadriflow(ctx)
        return None

    def cancel(self, ctx):
        if self.job:
            self.job.cancel()
            self.job = None

    def summary(self, ctx):
        return "Retopo (%s): %d quads (target %d)" % (
            ctx.flags.get("engine_used", "?"),
            ctx.flags.get("actual_quads", 0),
            ctx.settings["quads"])

    # -- engines ---------------------------------------------------------------

    def _sharp_edge(self, ctx):
        adv = ctx.settings["advanced"]
        # ORGANIC disables feature-edge preservation (180 deg = nothing sharp)
        return adv["sharp_edge"] if adv["model_type"] == "HARDSURFACE" else 180.0

    def _adopt_autoremesher(self, ctx, job):
        name = naming.next_mesh_name(ctx.base, bpy.data.objects.keys())
        mesh = job.result(name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        ctx.track_object(obj)
        self._install_result(ctx, obj, "AutoRemesher")

    def _quadriflow(self, ctx):
        work = ctx.obj(ctx.work_name)
        if work is None:
            raise StageError("Retopo: work mesh is missing")
        name = naming.next_mesh_name(ctx.base, bpy.data.objects.keys())
        dup = work.copy()
        dup.data = work.data.copy()
        dup.name = name
        dup.data.name = name
        bpy.context.scene.collection.objects.link(dup)
        ctx.track_object(dup)
        adv = ctx.settings["advanced"]
        try:
            with bpy.context.temp_override(object=dup, active_object=dup,
                                           selected_objects=[dup],
                                           selected_editable_objects=[dup]):
                result = bpy.ops.object.quadriflow_remesh(
                    target_faces=int(ctx.settings["quads"]),
                    mode="FACES",
                    use_mesh_symmetry=False,
                    use_preserve_sharp=(adv["model_type"] == "HARDSURFACE"),
                    seed=0)
        except RuntimeError as exc:
            raise StageError("QuadriFlow remesh failed: %s" % exc)
        if result != {"FINISHED"} or len(dup.data.polygons) == 0:
            raise StageError(
                "QuadriFlow produced no output — the merged mesh may not be manifold")
        self._install_result(ctx, dup, "QuadriFlow")

    # -- result / scaffold handling ---------------------------------------------

    def _install_result(self, ctx, obj, engine_name):
        work = ctx.obj(ctx.work_name)
        if work is not None:
            obj.matrix_world = work.matrix_world.copy()
            self._remove_work(ctx, work)
        elif self._work_matrix is not None:
            obj.matrix_world = self._work_matrix.copy()
        ctx.result_name = obj.name
        ctx.flags["engine_used"] = engine_name
        ctx.flags["actual_quads"] = len(obj.data.polygons)
        ctx.log("retopo: %s produced %d polygons"
                % (engine_name, ctx.flags["actual_quads"]))

    def _remove_work(self, ctx, work):
        # Keep the mesh datablock (0 users, purged on save) so a gate Redo
        # can resurrect the scaffold; the object itself goes away now.
        self._work_mesh_name = work.data.name
        self._work_matrix = work.matrix_world.copy()
        bpy.data.objects.remove(work)

    def _resurrect_work(self, ctx):
        if not (ctx.work_name and self._work_mesh_name):
            return None
        mesh = bpy.data.meshes.get(self._work_mesh_name)
        if mesh is None:
            return None
        work = bpy.data.objects.new(ctx.work_name, mesh)
        bpy.context.scene.collection.objects.link(work)
        if self._work_matrix is not None:
            work.matrix_world = self._work_matrix.copy()
        ctx.track_object(work)
        ctx.log("retopo: resurrected work scaffold for redo")
        return work


classes = ()
