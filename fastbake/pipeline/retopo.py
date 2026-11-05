# Retopology stage: AutoRemesher subprocess (preferred) with a blocking
# QuadriFlow fallback. Async — poll() drives the subprocess; QuadriFlow runs
# synchronously (seconds-scale, allowed to block).

import math

import bmesh
import bpy
import numpy as np
from mathutils import Matrix

from .. import naming
from ..engine import autoremesher, core
from .context import Stage, StageError

# Quad remeshers' peak memory scales with INPUT face count, not the quad
# target: QuadriFlow's orientation/position field solve and AutoRemesher's
# OBJ round-trip both blow up on raw multi-million-poly input. A 2M-face mesh
# can exhaust RAM long before the target is reached — and QuadriFlow runs
# synchronously, so it takes the whole UI down with it. The merge scaffold is
# only bounded on the voxel path (a clean single manifold is reused at full
# resolution), so cap the density here, before either engine runs.
MAX_RETOPO_INPUT_FACES = 300_000

# Both engines only APPROXIMATE the requested quad count (AutoRemesher solves
# for a field, QuadriFlow for a target edge length). Plan section 5 stage 2:
# read the actual count back and, if it misses the budget by more than 25 %,
# retry ONCE with the target scaled by the inverse of the observed ratio.
QUAD_TOLERANCE = 0.25
# AutoRemesher can only place quads where the input has vertices: a boolean
# (fast-path) scaffold keeps the sources' original tessellation, so a cube is
# still 12 triangles and the field solver has nothing to work with on it. The
# quad count then collapses far below the target (measured 2026-09-17: 160
# quads for a 1200 target on a 3.8k-triangle sphere+cube union, nearly all of
# them on the sphere). The voxel path already produces ~40 faces per target
# quad; give the out-of-process engines the same uniform density here.
# QuadriFlow is left alone — it refines flat regions on its own.
ASYNC_MIN_TRIS_PER_QUAD = 8.0        # below this the scaffold is re-voxelised
ASYNC_TARGET_TRIS_PER_QUAD = 30.0    # density aimed for, capped by the input ceiling
MIN_QUAD_TARGET = 16
MAX_QUAD_TARGET = 2_000_000


# Share of non-finite vertices an engine result may lose and still be used.
# QuadriFlow emits NaN positions for tiny disconnected pieces (the field solve
# has nothing to work with on a speck of a few triangles) — those pieces are
# dropped; anything wider than this is a broken result.
NONFINITE_DROP_FRACTION = 0.02


def _nonfinite_vertex_mask(me):
    n = len(me.vertices)
    if n == 0:
        return np.zeros(0, dtype=bool)
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    return ~np.isfinite(co.reshape(-1, 3)).all(axis=1)


def _mesh_is_finite(me):
    """False when any vertex coordinate is NaN/inf; such a mesh must never
    become the result — the UV stage would solve on NaN and the bake would
    go black."""
    if len(me.vertices) == 0:
        return False
    return not bool(_nonfinite_vertex_mask(me).any())


def _drop_nonfinite_vertices(me):
    """Delete NaN/inf vertices together with their faces. Returns the count."""
    mask = _nonfinite_vertex_mask(me)
    bad = int(mask.sum())
    if bad == 0:
        return 0
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        bm.verts.ensure_lookup_table()
        doomed = [bm.verts[i] for i in np.nonzero(mask)[0]]
        bmesh.ops.delete(bm, geom=doomed, context="VERTS")
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()
    return bad


def _shade_smooth(me):
    """Smooth-shade the low-poly before anything is baked onto it.

    Both engines hand back flat faces (QuadriFlow sets no smooth flag, the
    core's arrays carry none). The tangent-space normal map is baked against
    the low-poly's shading normals, so a flat result bakes every facet into
    the map and shows faceting and seams as soon as it is smoothed, and the
    bake rays leave each flat face along its own normal, leaving cracks at the
    edges. Smooth shading fixes both; hard edges stay in the normal map."""
    if hasattr(me, "shade_smooth"):        # 4.1+: also clears sharp_edge
        me.shade_smooth()
    else:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()


def _finite_or_trimmed(ctx, me, engine):
    """True when `me` is usable: finite, or made finite by dropping a tiny
    share of non-finite vertices (logged + reported). False = broken."""
    total = len(me.vertices)
    if total == 0:
        return False
    bad = int(_nonfinite_vertex_mask(me).sum())
    if bad == 0:
        return True
    if bad > max(1, int(total * NONFINITE_DROP_FRACTION)):
        ctx.log("retopo: %s output has %d non-finite vertices (%.1f%%) — unusable"
                % (engine, bad, 100.0 * bad / total))
        return False
    _drop_nonfinite_vertices(me)
    ctx.log("retopo: %s output: dropped %d non-finite vertices (degenerate specks)"
            % (engine, bad))
    ctx.notice("%s produced %d unusable vertices on tiny parts — they were "
               "dropped from the result" % (engine, bad))
    return len(me.polygons) > 0


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
        self._retried = False   # the one quad-count correction pass is used up
        self._first_mesh_name = None   # first async result kept across the retry

    def start(self, ctx):
        self.job = None
        self._retried = False
        self._drop_first_result()
        work = ctx.obj(ctx.work_name)
        if work is None:
            work = self._resurrect_work(ctx)
        if work is None:
            raise StageError("Retopo: work mesh is missing")

        self._cap_input_density(ctx, work)

        target = int(ctx.settings["quads"])
        engine = ctx.settings["advanced"]["engine"]
        if engine in ("AUTO", "AUTOREMESHER"):
            if core.core_available() or autoremesher.find_binary():
                self._densify_for_async(ctx, work, target)
            if self._launch_async(ctx, work, target):
                return
            if engine == "AUTOREMESHER":
                ctx.notice("AutoRemesher not available — used QuadriFlow instead")
            else:
                ctx.log("AutoRemesher not found; using QuadriFlow")
        self._quadriflow(ctx, target)

    def _launch_async(self, ctx, work, target):
        """Start an out-of-process engine for `target` quads.

        True when a subprocess is running and poll() will drive it. Order:
        the bundled autoremesher_core wheel (no external dependency, no Qt,
        crash-isolated, real progress), then a user-supplied AutoRemesher
        binary. Also used to relaunch for the quad-count retry.
        """
        hard = ctx.settings["advanced"]["model_type"] == "HARDSURFACE"
        if core.core_available():
            self.job = core.CoreRemeshJob(ctx)
            try:
                self.job.start(work, target, self._sharp_edge(ctx),
                               hard_surface=hard)
                return True
            except (OSError, ValueError) as exc:
                ctx.log("bundled AutoRemesher core launch failed: %s" % exc)
                self.job = None
        exe = autoremesher.find_binary()
        if exe:
            self.job = autoremesher.AutoRemesherJob(exe, ctx)
            try:
                self.job.start(work, target, self._sharp_edge(ctx))
                return True
            except OSError as exc:
                ctx.log("autoremesher launch failed: %s" % exc)
                self.job = None
        return False

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
            self._quadriflow(ctx, int(ctx.settings["quads"]))
            return None

        # subprocess finished
        job, self.job = self.job, None
        job.close()
        if job.succeeded():
            try:
                if self._adopt_autoremesher(ctx, job):
                    # a quad-count retry is running: stay in this stage
                    return self.job.progress()
                return None
            except (ValueError, OSError, IndexError) as exc:
                ctx.log("autoremesher output unusable: %s" % exc)
        else:
            code = job.proc.returncode if job.proc else "?"
            ctx.log("autoremesher failed (exit %s): %s"
                    % (code, job.error_excerpt()))
        if self._first_mesh_name and self._install_first_result(ctx):
            ctx.notice("AutoRemesher's corrective pass failed — kept its first result")
            return None
        ctx.notice("AutoRemesher failed — fell back to QuadriFlow")
        self._quadriflow(ctx, int(ctx.settings["quads"]))
        return None

    def cancel(self, ctx):
        if self.job:
            self.job.cancel()
            self.job = None
        self._drop_first_result()

    # -- quad-count retry bookkeeping ---------------------------------------------

    def _drop_first_result(self):
        first = (bpy.data.meshes.get((self._first_mesh_name, None))
                 if self._first_mesh_name else None)
        if first is not None and first.users == 0:
            bpy.data.meshes.remove(first)
        self._first_mesh_name = None

    def _closer_of(self, ctx, mesh):
        """The retry is adopted only when it lands closer to the target than
        the first attempt; otherwise the first result is kept (the correction
        is a heuristic and can overshoot)."""
        first = (bpy.data.meshes.get((self._first_mesh_name, None))
                 if self._first_mesh_name else None)
        self._first_mesh_name = None
        if first is None or first is mesh:
            return mesh
        target = int(ctx.settings["quads"])
        if abs(len(first.polygons) - target) < abs(len(mesh.polygons) - target):
            ctx.log("retopo: retry landed farther from the target (%d vs %d "
                    "quads) — keeping the first result"
                    % (len(mesh.polygons), len(first.polygons)))
            bpy.data.meshes.remove(mesh)
            return first
        bpy.data.meshes.remove(first)
        return mesh

    def _install_first_result(self, ctx):
        first = (bpy.data.meshes.get((self._first_mesh_name, None))
                 if self._first_mesh_name else None)
        self._first_mesh_name = None
        if first is None:
            return False
        ctx.log("retopo: quad-count retry failed — keeping the first "
                "AutoRemesher result (%d polygons)" % len(first.polygons))
        name = naming.next_mesh_name(ctx.base, self._existing_names())
        first.name = name
        obj = bpy.data.objects.new(name, first)
        bpy.context.scene.collection.objects.link(obj)
        ctx.track_object(obj)
        self._install_result(ctx, obj, "AutoRemesher")
        return True

    def summary(self, ctx):
        return "Retopo (%s): %d quads (target %d)" % (
            ctx.flags.get("engine_used", "?"),
            ctx.flags.get("actual_quads", 0),
            ctx.settings["quads"])

    # -- engines ---------------------------------------------------------------

    def _cap_input_density(self, ctx, work):
        """Collapse-decimate an over-dense scaffold to a safe face ceiling
        before remeshing. Both engines' memory scales with input density;
        output quality is unaffected because the remesher rebuilds topology to
        the quad target regardless. Best-effort: on failure we log and hand the
        full-resolution mesh over (the prior behaviour)."""
        from .merge import _tri_count, weld_short_edges
        tris = _tri_count(work.data)
        if tris <= MAX_RETOPO_INPUT_FACES:
            return
        mod = work.modifiers.new("FB_RetopoDecimate", "DECIMATE")
        mod.decimate_type = "COLLAPSE"
        mod.ratio = MAX_RETOPO_INPUT_FACES / tris
        mod.use_collapse_triangulate = True
        mod_name = mod.name
        try:
            with bpy.context.temp_override(object=work, active_object=work,
                                           selected_objects=[work]):
                bpy.ops.object.modifier_apply(modifier=mod_name)
        except RuntimeError as exc:
            leftover = work.modifiers.get(mod_name)
            if leftover is not None:
                work.modifiers.remove(leftover)
            ctx.log("retopo: input density cap failed (%s) — remeshing at "
                    "full resolution" % exc)
            return
        # Collapse can leave sub-1e-4 slivers that QuadriFlow rejects — weld
        # them out just as the merge voxel path does.
        weld_short_edges(work.data)
        after = _tri_count(work.data)
        ctx.notice("Source was very dense (%s triangles) — reduced to %s before "
                   "retopology to stay within memory."
                   % ("{:,}".format(tris), "{:,}".format(after)))
        ctx.log("retopo: capped input density %s -> %s tris"
                % ("{:,}".format(tris), "{:,}".format(after)))

    def _densify_for_async(self, ctx, work, target):
        """Re-voxelise a too-coarse scaffold before an AutoRemesher run.

        Best-effort like _cap_input_density: on any failure the scaffold is
        handed over as it is. Works in the scaffold's LOCAL space (the merge
        stage may have scaled it up), so the voxel floor is the local one."""
        from .merge import (QF_EDGE_EPS, SAFE_LOCAL_VOXEL, _surface_area,
                            _tri_count, weld_short_edges)
        tris = _tri_count(work.data)
        want = int(min(target * ASYNC_TARGET_TRIS_PER_QUAD,
                       MAX_RETOPO_INPUT_FACES * 0.8))
        if tris >= min(target * ASYNC_MIN_TRIS_PER_QUAD, want):
            return
        area = _surface_area(work.data)
        if area <= 0.0:
            return
        # dual contouring yields roughly area / voxel^2 quads = 2x triangles
        voxel = max(math.sqrt(2.0 * area / want), SAFE_LOCAL_VOXEL)
        try:
            work.data.remesh_voxel_size = voxel
            work.data.remesh_voxel_adaptivity = 0.0
            with bpy.context.temp_override(object=work, active_object=work,
                                           selected_objects=[work]):
                bpy.ops.object.voxel_remesh()
        except RuntimeError as exc:
            ctx.log("retopo: densify for AutoRemesher failed (%s) — remeshing "
                    "the %d-tri scaffold as is" % (exc, tris))
            return
        weld_short_edges(work.data, dist=min(QF_EDGE_EPS * 2.0, voxel * 0.25))
        ctx.log("retopo: densified the %d-tri scaffold to %d tris (voxel %.4g) "
                "for AutoRemesher" % (tris, _tri_count(work.data), voxel))

    def _sharp_edge(self, ctx):
        adv = ctx.settings["advanced"]
        # ORGANIC disables feature-edge preservation (180 deg = nothing sharp)
        return adv["sharp_edge"] if adv["model_type"] == "HARDSURFACE" else 180.0

    def _existing_names(self):
        """Object names a new result could actually collide with. Linked
        (library) IDs live in their own namespace and never block a local
        name, so they are ignored (contract decision 10)."""
        return [ob.name for ob in bpy.data.objects if ob.library is None]

    def _retry_target(self, ctx, actual):
        """Corrected quad target, or None when the result is good enough.

        Plan section 5 stage 2: the engines' output count is approximate; if
        it misses the budget by more than 25 %, ask once more with the target
        scaled by the inverse of the observed ratio."""
        if self._retried:
            return None
        target = int(ctx.settings["quads"])
        if target <= 0 or actual <= 0:
            return None
        miss = abs(actual - target) / float(target)
        if miss <= QUAD_TOLERANCE:
            return None
        scaled = int(round(target * (float(target) / actual)))
        scaled = max(MIN_QUAD_TARGET, min(scaled, MAX_QUAD_TARGET))
        if scaled == target:
            return None
        self._retried = True
        ctx.log("retopo: %d quads vs target %d (off by %.0f%%) — one retry at "
                "target %d" % (actual, target, miss * 100.0, scaled))
        return scaled

    def _adopt_autoremesher(self, ctx, job):
        """Turn the finished job's output into the result object.

        Returns True when the quad count was so far off that a corrective
        retry was launched instead — poll() then keeps driving that job."""
        name = naming.next_mesh_name(ctx.base, self._existing_names())
        mesh = job.result(name)
        scaled = self._retry_target(ctx, len(mesh.polygons))
        if scaled is not None:
            work = ctx.obj(ctx.work_name)
            # Relaunch only the async engines: QuadriFlow does its own
            # correction pass in _requad, and a failed relaunch here must not
            # cost us the result we already have.
            if work is not None and self._launch_async(ctx, work, scaled):
                self._first_mesh_name = mesh.name   # kept: adopted if closer
                return True
        mesh = self._closer_of(ctx, mesh)
        mesh.name = name
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        ctx.track_object(obj)
        self._install_result(ctx, obj, "AutoRemesher")
        return False

    def _quadriflow(self, ctx, target=None):
        work = ctx.obj(ctx.work_name)
        if work is None:
            raise StageError("Retopo: work mesh is missing")
        target = int(ctx.settings["quads"] if target is None else target)
        name = naming.next_mesh_name(ctx.base, self._existing_names())
        dup = work.copy()
        dup.data = work.data.copy()
        dup.name = name
        dup.data.name = name
        bpy.context.scene.collection.objects.link(dup)
        ctx.track_object(dup)
        adv = ctx.settings["advanced"]
        pristine = dup.data.copy()  # kept for the voxel-retry restore
        fed = None                  # the mesh the last QuadriFlow call consumed
        try:
            for attempt in ("direct", "voxel-retry"):
                if attempt == "voxel-retry":
                    # QuadriFlow refused the scaffold: force it through an
                    # OpenVDB voxel pass (manifold by construction), retry once.
                    old = dup.data
                    dup.data = pristine.copy()
                    if old.users == 0:
                        bpy.data.meshes.remove(old)
                    from .merge import (QF_EDGE_EPS, SAFE_LOCAL_VOXEL,
                                        _mesh_bbox_diag, weld_short_edges)
                    # LOCAL-space diagonal: both the voxel size and
                    # QuadriFlow's 1e-4 epsilon are local-space quantities,
                    # and the merge stage may have handed us a scaffold that
                    # is deliberately scaled up (merge._normalise_scale).
                    diag = _mesh_bbox_diag(dup.data) or 1.0
                    # Keep the voxel size safely above QuadriFlow's edge
                    # epsilon: a finer grid on small objects would be gutted
                    # by the weld below (and rejected by QuadriFlow anyway).
                    voxel = max(diag / 300.0, SAFE_LOCAL_VOXEL)
                    dup.data.remesh_voxel_size = voxel
                    dup.data.remesh_voxel_adaptivity = 0.0
                    with bpy.context.temp_override(object=dup, active_object=dup,
                                                   selected_objects=[dup]):
                        bpy.ops.object.voxel_remesh()
                    weld_short_edges(dup.data,
                                     dist=min(QF_EDGE_EPS * 2.0, voxel * 0.25))
                    ctx.log("retopo: voxel-retry pass before QuadriFlow "
                            "(%d faces)" % len(dup.data.polygons))
                if fed is not None and fed.users == 0:
                    bpy.data.meshes.remove(fed)
                fed = dup.data.copy()   # what this call is about to consume
                try:
                    with bpy.context.temp_override(object=dup, active_object=dup,
                                                   selected_objects=[dup],
                                                   selected_editable_objects=[dup]):
                        result = bpy.ops.object.quadriflow_remesh(
                            target_faces=target,
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
                    if _finite_or_trimmed(ctx, dup.data, "QuadriFlow"):
                        break
                    if attempt == "voxel-retry":
                        raise StageError("QuadriFlow produced non-finite geometry "
                                         "— try a higher Merge detail setting")
                    ctx.log("retopo: QuadriFlow output has non-finite vertices, "
                            "retrying via voxel")
                    continue
                if attempt == "voxel-retry":
                    raise StageError("QuadriFlow produced no output — the merged "
                                     "mesh may not be manifold")
                ctx.log("retopo: QuadriFlow produced no output, retrying via voxel")
            self._requad(ctx, dup, fed, adv)
        finally:
            if pristine.users == 0:
                bpy.data.meshes.remove(pristine)
            if fed is not None and fed.users == 0:
                bpy.data.meshes.remove(fed)
        dup.data.name = name
        self._install_result(ctx, dup, "QuadriFlow")

    def _requad(self, ctx, dup, fed, adv):
        """One corrective QuadriFlow pass when the quad count missed the
        budget by more than 25 % (plan section 5 stage 2). Runs on a copy of
        the very mesh the first call consumed and keeps whichever result lands
        closer to the target, so a retry can never make things worse."""
        target = int(ctx.settings["quads"])
        scaled = self._retry_target(ctx, len(dup.data.polygons))
        if scaled is None or fed is None:
            return
        first = dup.data
        first_err = abs(len(first.polygons) - target)
        dup.data = fed.copy()
        better = False
        try:
            with bpy.context.temp_override(object=dup, active_object=dup,
                                           selected_objects=[dup],
                                           selected_editable_objects=[dup]):
                result = bpy.ops.object.quadriflow_remesh(
                    target_faces=scaled,
                    mode="FACES",
                    use_mesh_symmetry=False,
                    use_preserve_sharp=(adv["model_type"] == "HARDSURFACE"),
                    seed=0)
            better = (result == {"FINISHED"} and len(dup.data.polygons) > 0
                      and abs(len(dup.data.polygons) - target) < first_err)
        except RuntimeError as exc:
            ctx.log("retopo: quad-count retry failed (%s) — keeping the first "
                    "result" % exc)
        retried = dup.data
        if better:
            ctx.log("retopo: quad-count retry %d -> %d quads (target %d)"
                    % (len(first.polygons), len(retried.polygons), target))
            if first.users == 0:
                bpy.data.meshes.remove(first)
        else:
            dup.data = first
            if retried.users == 0:
                bpy.data.meshes.remove(retried)

    # -- result / scaffold handling ---------------------------------------------

    def _install_result(self, ctx, obj, engine_name):
        if not _finite_or_trimmed(ctx, obj.data, engine_name):
            raise StageError("%s produced non-finite geometry — try the other "
                             "retopo engine or a higher Merge detail setting"
                             % engine_name)
        work = ctx.obj(ctx.work_name)
        matrix = None
        if work is not None:
            matrix = work.matrix_world.copy()
            self._remove_work(ctx, work)
        elif self._work_matrix is not None:
            matrix = self._work_matrix.copy()
        if matrix is not None:
            # The scaffold may have been merged at an internal scale
            # (merge._normalise_scale). Bake the transform into the vertices
            # rather than leaving it on the object: every downstream stage —
            # bake cage/ray distances, MikkTSpace tangents, UV area — assumes
            # the result sits at unit scale in world space.
            obj.data.transform(matrix)
            obj.matrix_world = Matrix.Identity(4)
        _shade_smooth(obj.data)
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
