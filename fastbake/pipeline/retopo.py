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
            if not self.job.timed_out():
                return self.job.progress()
            # Watchdog: the binary is hung (no exit long past the time
            # estimate, e.g. Qt blocked on a dead DISPLAY) — kill it and
            # fall back so headless runs cannot spin forever.
            job, self.job = self.job, None
            job.cancel()
            ctx.log("autoremesher watchdog: killed hung process")
            ctx.notice("AutoRemesher timed out — fell back to QuadriFlow")
            self._quadriflow(ctx)
            return None

        # subprocess finished
        job, self.job = self.job, None
        job.close()
        if job.succeeded():
            try:
                self._adopt_autoremesher(ctx, job)
                return None
            except (ValueError, OSError, IndexError) as exc:
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
        pristine = dup.data.copy()  # kept for the voxel-retry restore
        try:
            for attempt in ("direct", "voxel-retry"):
                if attempt == "voxel-retry":
                    # QuadriFlow refused the scaffold: force it through an
                    # OpenVDB voxel pass (manifold by construction), retry once.
                    old = dup.data
                    dup.data = pristine.copy()
                    if old.users == 0:
                        bpy.data.meshes.remove(old)
                    from .merge import QF_EDGE_EPS, weld_short_edges
                    diag = max(dup.dimensions[:]) or 1.0
                    # Keep the voxel size safely above QuadriFlow's edge
                    # epsilon: a finer grid on small objects would be gutted
                    # by the weld below (and rejected by QuadriFlow anyway).
                    voxel = max(diag / 300.0, QF_EDGE_EPS * 20)
                    dup.data.remesh_voxel_size = voxel
                    dup.data.remesh_voxel_adaptivity = 0.0
                    with bpy.context.temp_override(object=dup, active_object=dup,
                                                   selected_objects=[dup]):
                        bpy.ops.object.voxel_remesh()
                    weld_short_edges(dup.data,
                                     dist=min(QF_EDGE_EPS * 2.0, voxel * 0.25))
                    ctx.log("retopo: voxel-retry pass before QuadriFlow "
                            "(%d faces)" % len(dup.data.polygons))
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
                    if attempt == "voxel-retry":
                        raise StageError("QuadriFlow remesh failed: %s" % exc)
                    ctx.log("retopo: QuadriFlow raised (%s), retrying via voxel" % exc)
                    continue
                if result == {"FINISHED"} and len(dup.data.polygons) > 0:
                    break
                if attempt == "voxel-retry":
                    raise StageError("QuadriFlow produced no output — the merged "
                                     "mesh may not be manifold")
                ctx.log("retopo: QuadriFlow produced no output, retrying via voxel")
        finally:
            if pristine.users == 0:
                bpy.data.meshes.remove(pristine)
        dup.data.name = name
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
        # rollback_to cannot reach the mesh once the object is gone, so an
        # undo hook frees it on cancel/rollback; ctx.flags records the name
        # so the final cleanup can free it on success too.
        mesh_name = work.data.name
        if mesh_name != self._work_mesh_name:
            def _free_kept_mesh():
                mesh = bpy.data.meshes.get(mesh_name)
                if mesh is not None and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
            ctx.undo_hooks.append(_free_kept_mesh)
        self._work_mesh_name = mesh_name
        self._work_matrix = work.matrix_world.copy()
        ctx.flags["retopo_kept_mesh"] = mesh_name
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
