# Merge stage: snapshot the selected source meshes through the evaluated
# depsgraph, join the copies into one work object and turn it into a
# watertight manifold scaffold for the retopo stage.
#
# Path selection (see dev/CONTRACTS.md and the watertight-merge research):
# * single clean manifold source           -> keep the duplicate as-is
# * several clean closed shells            -> boolean Exact self-union
#   (empty-collection operand; feature-preserving)
# * anything open / non-manifold / failed  -> solidify open copies, voxel
#   remesh, manual loose-part cleanup, validate, retry at x0.7 then x1.5
#
# NOTE: the Manifold boolean solver refuses an empty operand collection
# ("Cannot execute, non-exact solver and empty collection", verified on
# 5.1.2) and silently leaves the mesh unchanged, so the self-union trick
# only works with the Exact solver.

import bpy
import bmesh
import numpy as np

from .. import compat
from .context import Stage, StageError

# "Merge detail" preset -> bbox-diagonal divisor for the voxel size.
VOXEL_DIVISOR = {"LOW": 150.0, "MED": 300.0, "HIGH": 600.0}
MAX_VOXEL_DIVISOR = 1500.0   # memory clamp: never finer than diagonal/1500
SOLIDIFY_FACTOR = 2.5        # open-sheet thickness, in voxel sizes
LOOSE_PART_FRACTION = 0.02   # drop parts with bbox diagonal < 2% of the main part


# --- mesh census (numpy over foreach_get, fast on multi-million tris) --------

def _diagnose(me):
    """Edge-manifoldness census of a mesh.

    boundary: edges with exactly one face; bad: wire edges or edges with 3+
    faces; flipped: two-face edges whose faces wind the same way (inconsistent
    normals). Watertight manifold iff boundary == bad == flipped == 0.
    """
    n_edges = len(me.edges)
    n_loops = len(me.loops)
    result = {"boundary": 0, "bad": 0, "flipped": 0, "faces": len(me.polygons)}
    if n_edges == 0:
        return result
    ev = np.empty(n_edges * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    le = np.empty(n_loops, dtype=np.int64)
    me.loops.foreach_get("edge_index", le)
    lv = np.empty(n_loops, dtype=np.int64)
    me.loops.foreach_get("vertex_index", lv)
    count = np.bincount(le, minlength=n_edges)
    result["boundary"] = int((count == 1).sum())
    result["bad"] = int((count == 0).sum() + (count > 2).sum())
    # A loop's edge starts at the loop's own vertex. Winding is consistent
    # when the two faces sharing an edge traverse it in opposite directions.
    same = (lv == ev[le, 0]).astype(np.int64)
    dirsum = np.bincount(le, weights=same, minlength=n_edges).astype(np.int64)
    result["flipped"] = int(((count == 2) & (dirsum != 1)).sum())
    return result


# QuadriFlow rejects meshes containing any edge whose endpoints agree within
# 1e-4 on every axis ("zero length edges", mesh_is_manifold_consistent in
# object_remesh.cc). Boolean intersections routinely emit such slivers.
QF_EDGE_EPS = 1e-4


def _short_edge_count(me, eps=QF_EDGE_EPS * 1.5):
    n = len(me.edges)
    if n == 0:
        return 0
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ev = np.empty(n * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    # compare_v3v3 semantics: "zero length" iff max per-axis delta < eps
    d = np.abs(co[ev[:, 0]] - co[ev[:, 1]]).max(axis=1)
    return int((d < eps).sum())


def weld_short_edges(me, dist=QF_EDGE_EPS * 2.0):
    """Collapse sub-threshold edges + degenerate faces. Returns verts removed."""
    # the gating eps scales with dist (default keeps the padded 1.5e-4 check)
    if _short_edge_count(me, eps=dist * 0.75) == 0:
        return 0
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        before = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=dist)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges[:], dist=dist)
        removed = before - len(bm.verts)
        if removed:
            bm.to_mesh(me)
            me.update()
        return removed
    finally:
        bm.free()


def _nonmanifold_vert_count(me):
    """Bowtie/junction vertices — invisible to the edge census but fatal for
    quad remeshers (QuadriFlow refuses vertex-non-manifold input)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        return sum(1 for v in bm.verts if not v.is_manifold)
    finally:
        bm.free()


def _tri_count(me):
    n = len(me.polygons)
    if n == 0:
        return 0
    lt = np.empty(n, dtype=np.int64)
    me.polygons.foreach_get("loop_total", lt)
    return int((lt - 2).sum())


def _mesh_bbox_diag(me):
    n = len(me.vertices)
    if n == 0:
        return 0.0
    co = np.empty(n * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    return float(np.linalg.norm(co.max(0) - co.min(0)))


def _world_bbox_diag(objs):
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for ob in objs:
        n = len(ob.data.vertices)
        if n == 0:
            continue
        co = np.empty(n * 3, dtype=np.float32)
        ob.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3).astype(np.float64)
        m = np.array(ob.matrix_world)
        world = co @ m[:3, :3].T + m[:3, 3]
        lo = np.minimum(lo, world.min(0))
        hi = np.maximum(hi, world.max(0))
    if not np.isfinite(lo).all():
        return 0.0
    return float(np.linalg.norm(hi - lo))


# --- destructive helpers ------------------------------------------------------

def _join_into(work, others):
    """Join `others` into `work`; returns the absorbed object names.

    object.join removes the absorbed objects but orphans their mesh
    datablocks — those are freed here too.
    """
    absorbed = [ob.name for ob in others]
    meshes = [ob.data.name for ob in others]
    everything = [work] + list(others)
    with compat.temp_override(active_object=work, selected_objects=everything,
                              selected_editable_objects=everything):
        bpy.ops.object.join()
    for name in meshes:
        me = bpy.data.meshes.get(name)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    return absorbed


def _restore_mesh(obj, pristine):
    """Swap in a fresh copy of the pristine mesh, freeing the current one."""
    old = obj.data
    obj.data = pristine.copy()
    if old.users == 0:
        bpy.data.meshes.remove(old)


def _recalc_normals(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()


def _apply_solidify(ctx, obj, thickness):
    """Give an open sheet thickness so the voxel SDF can represent it."""
    mod = obj.modifiers.new("FastBake Solidify", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = 0.0  # grow evenly around the sheet
    mod_name = mod.name
    try:
        with compat.temp_override(object=obj, active_object=obj,
                                  selected_objects=[obj]):
            bpy.ops.object.modifier_apply(modifier=mod_name)
        ctx.log("merge: solidified open mesh %s (thickness %.5g)"
                % (obj.name, thickness))
    except RuntimeError as exc:
        ctx.log("merge: solidify on %s failed: %s" % (obj.name, exc))
        mod = obj.modifiers.get(mod_name)
        if mod is not None:
            obj.modifiers.remove(mod)


def _apply_self_union(ctx, obj, solver):
    """Boolean union of the object with an empty collection: dissolves the
    interior (self-intersecting) geometry of the joined shells in place."""
    col = bpy.data.collections.new("fastbake_union_operand")
    mod = obj.modifiers.new("FastBake Union", "BOOLEAN")
    mod.operation = "UNION"
    mod.operand_type = "COLLECTION"
    mod.collection = col
    mod.solver = solver
    if solver == "EXACT":
        mod.use_self = True
    mod_name = mod.name
    try:
        with compat.temp_override(object=obj, active_object=obj,
                                  selected_objects=[obj]):
            bpy.ops.object.modifier_apply(modifier=mod_name)
        return True
    except RuntimeError as exc:
        ctx.log("merge: boolean %s self-union failed: %s" % (solver, exc))
        mod = obj.modifiers.get(mod_name)
        if mod is not None:
            obj.modifiers.remove(mod)
        return False
    finally:
        bpy.data.collections.remove(col)


def _apply_voxel_remesh(obj, voxel_size):
    me = obj.data
    me.remesh_voxel_size = voxel_size
    me.remesh_voxel_adaptivity = 0.0  # AutoRemesher wants uniform density
    if hasattr(me, "use_remesh_fix_poles"):
        me.use_remesh_fix_poles = False
    with compat.temp_override(object=obj, active_object=obj,
                              selected_objects=[obj]):
        bpy.ops.object.voxel_remesh()


# --- the stage ----------------------------------------------------------------

class MergeStage(Stage):
    stage_id = "merge"
    label = "Merging"
    gate = True
    adjust_props = ("merge_detail",)

    _n_src = 0
    _tris = 0
    _defects = 0
    _path = "?"
    _notice = None

    def start(self, ctx):
        self._n_src = 0
        self._tris = 0
        self._defects = 0
        self._path = "?"
        # A gate Redo re-runs start() on this same instance: retract the
        # watertightness notice a previous attempt left in the report.
        if self._notice and self._notice in ctx.notices:
            ctx.notices.remove(self._notice)
        self._notice = None

        copies = self._snapshot_sources(ctx)
        self._n_src = len(copies)

        diag = _world_bbox_diag(copies)
        if diag <= 0.0:
            raise StageError("Nothing to merge — the selection has zero size")
        adv = ctx.settings["advanced"]
        divisor = VOXEL_DIVISOR.get(adv.get("merge_detail", "MED"), 300.0)
        voxel = max(diag / divisor, diag / MAX_VOXEL_DIVISOR)

        # Per-copy diagnosis before joining, so repairs stay local: recalc
        # inconsistent normals, give open sheets thickness for the SDF.
        boundary = bad = flipped = 0
        for dup in copies:
            d = _diagnose(dup.data)
            if d["flipped"]:
                _recalc_normals(dup.data)
                ctx.log("merge: recalculated inconsistent normals on " + dup.name)
                d = _diagnose(dup.data)
            if d["boundary"]:
                _apply_solidify(ctx, dup, SOLIDIFY_FACTOR * voxel)
            boundary += d["boundary"]
            bad += d["bad"]
            flipped += d["flipped"]

        work = self._join(ctx, copies)
        clean = not (boundary or bad or flipped)

        pristine = work.data.copy()
        try:
            if clean and self._n_src == 1:
                self._path = "reused as-is"
                ctx.log("merge: single manifold source, duplicate reused as-is")
            else:
                done = False
                if clean:
                    done = self._boolean_path(ctx, work, pristine)
                else:
                    ctx.log("merge: robust path (%d boundary, %d non-manifold, "
                            "%d flipped edges in sources)" % (boundary, bad, flipped))
                if not done:
                    self._voxel_path(ctx, work, pristine, voxel)
        finally:
            if pristine.users == 0:
                bpy.data.meshes.remove(pristine)

        work.data.name = work.name
        self._tris = _tri_count(work.data)
        ctx.work_name = work.name
        ctx.log("merge: scaffold %s — %d tris, %s" % (work.name, self._tris, self._path))

    def summary(self, ctx):
        noun = "mesh" if self._n_src == 1 else "meshes"
        if self._defects:
            quality = "scaffold with %d open edges" % self._defects
        else:
            quality = "watertight scaffold"
        return "Merged %d %s -> %s (%s tris, %s)" % (
            self._n_src, noun, quality, "{:,}".format(self._tris), self._path)

    # --- steps ----------------------------------------------------------------

    def _snapshot_sources(self, ctx):
        """Depsgraph-evaluated copies of the sources (modifiers applied),
        linked to the scene and tracked for rollback. Originals untouched."""
        deps = bpy.context.evaluated_depsgraph_get()
        scene_col = bpy.context.scene.collection
        copies = []
        for name in ctx.src_names:
            src = ctx.obj(name)
            if src is None or src.type != "MESH":
                continue
            evo = src.evaluated_get(deps)
            me = bpy.data.meshes.new_from_object(
                evo, preserve_all_data_layers=False, depsgraph=deps)
            if me is None or len(me.polygons) == 0:
                if me is not None:
                    bpy.data.meshes.remove(me)
                ctx.log("merge: skipping %s — no faces after evaluation" % name)
                continue
            me.materials.clear()  # the scaffold never renders; bake uses originals
            # Bake the world transform into the vertices so local space equals
            # world space: voxel size and solidify thickness are derived from
            # the world bbox but consumed in local units, and parent-inherited
            # scale survives ApplyTransformsStage.
            mw = evo.matrix_world.copy()
            me.transform(mw)
            if mw.is_negative:
                me.flip_normals()  # a mirrored transform inverts the winding
            dup = bpy.data.objects.new(name + "_fbtmp", me)  # identity matrix
            scene_col.objects.link(dup)
            ctx.track_object(dup)
            copies.append(dup)
        if not copies:
            raise StageError("Nothing to merge — the selected objects have no faces")
        return copies

    def _join(self, ctx, copies):
        work = copies[0]
        if len(copies) > 1:
            for name in _join_into(work, copies[1:]):
                if name in ctx.created_objects:
                    ctx.created_objects.remove(name)
        # rename into place, keeping the rollback tracking in sync
        if work.name in ctx.created_objects:
            ctx.created_objects.remove(work.name)
        work.name = ctx.base + "_fbwork"
        ctx.track_object(work)
        return work

    def _boolean_path(self, ctx, work, pristine):
        """Feature-preserving fast path for clean closed shells."""
        # prefer_manifold=False: the Manifold solver cannot self-union with an
        # empty operand collection (silent no-op, see module header).
        solver = compat.boolean_solver(False)
        if not _apply_self_union(ctx, work, solver):
            _restore_mesh(work, pristine)
            return False
        welded = weld_short_edges(work.data)
        if welded:
            ctx.log("merge: welded %d sliver vertices after self-union" % welded)
        d = _diagnose(work.data)
        nm_verts = _nonmanifold_vert_count(work.data)
        # recheck at QuadriFlow's real 1e-4 rejection threshold — the padded
        # default would discard results QuadriFlow actually accepts
        if _short_edge_count(work.data, eps=QF_EDGE_EPS):
            ctx.log("merge: self-union left sub-1e-4 edges — falling back to voxel")
            _restore_mesh(work, pristine)
            return False
        if d["faces"] and not (d["boundary"] or d["bad"] or d["flipped"] or nm_verts):
            self._path = "boolean path"
            ctx.log("merge: %s self-union succeeded (%d faces)"
                    % (solver.lower(), d["faces"]))
            return True
        ctx.log("merge: self-union result not clean (%d boundary, "
                "%d non-manifold edges, %d non-manifold verts) — falling "
                "back to voxel" % (d["boundary"], d["bad"], nm_verts))
        _restore_mesh(work, pristine)
        return False

    def _voxel_path(self, ctx, work, pristine, voxel):
        """OpenVDB voxel remesh with a retry ladder (x1.0, x0.7, x1.5)."""
        attempts = (voxel, voxel * 0.7, voxel * 1.5)
        scores = []    # defect count per attempt, None = unusable
        current = None  # index of the attempt the work mesh currently holds
        for i, vs in enumerate(attempts):
            _restore_mesh(work, pristine)
            current = None
            try:
                _apply_voxel_remesh(work, vs)
            except RuntimeError as exc:
                ctx.log("merge: voxel remesh at %.5g failed: %s" % (vs, exc))
                scores.append(None)
                continue
            current = i
            # clamp the weld distance well below the voxel edge length, or a
            # fine grid (small real-world-scale objects) would collapse
            weld_short_edges(work.data, dist=min(QF_EDGE_EPS * 2.0, vs * 0.25))
            dropped = self._drop_loose_parts(ctx, work)
            d = _diagnose(work.data)
            defects = (d["boundary"] + d["bad"]) if d["faces"] else None
            scores.append(defects)
            ctx.log("merge: voxel %.5g -> %d faces, %s defect edges, "
                    "%d loose parts dropped"
                    % (vs, d["faces"],
                       "no" if defects == 0 else defects, dropped))
            if defects == 0:
                self._path = "voxel path"
                self._defects = 0
                return
        usable = [(s, i) for i, s in enumerate(scores) if s is not None]
        if not usable:
            raise StageError("Merge failed: voxel remesh produced no usable "
                             "geometry — try a higher Merge detail setting")
        best_score, best_i = min(usable)
        if best_i != current:
            _restore_mesh(work, pristine)
            _apply_voxel_remesh(work, attempts[best_i])
            weld_short_edges(work.data,
                             dist=min(QF_EDGE_EPS * 2.0, attempts[best_i] * 0.25))
            self._drop_loose_parts(ctx, work)
        self._path = "voxel path"
        self._defects = best_score
        self._notice = ("Merged scaffold is not fully watertight (%d problem "
                        "edges) — retopology may struggle" % best_score)
        ctx.notice(self._notice)

    def _drop_loose_parts(self, ctx, work):
        """Voxel mode has no Remove Disconnected option: split into loose
        parts, drop fragments with a bbox diagonal under 2% of the main part,
        rejoin the rest. Returns the number of dropped fragments."""
        # mesh.separate needs the real view-layer selection/active — save the
        # user's and put it back, mirroring how src_visibility is handled
        sel_names = [ob.name for ob in bpy.context.selected_objects]
        active = bpy.context.view_layer.objects.active
        active_name = active.name if active else None
        try:
            return self._separate_and_drop(ctx, work)
        finally:
            for ob in list(bpy.context.selected_objects):
                ob.select_set(False)
            for name in sel_names:
                ob = bpy.data.objects.get(name)
                if ob is not None:
                    try:
                        ob.select_set(True)
                    except RuntimeError:
                        pass  # no longer in the view layer
            bpy.context.view_layer.objects.active = (
                bpy.data.objects.get(active_name) if active_name else None)

    def _separate_and_drop(self, ctx, work):
        before = set(bpy.data.objects.keys())
        for ob in list(bpy.context.selected_objects):
            ob.select_set(False)
        work.select_set(True)
        bpy.context.view_layer.objects.active = work
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            bpy.ops.mesh.separate(type="LOOSE")
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        part_names = [n for n in bpy.data.objects.keys() if n not in before]
        if not part_names:
            return 0
        parts = [work] + [bpy.data.objects[n] for n in part_names]
        for p in parts[1:]:
            ctx.track_object(p)
        diags = [_mesh_bbox_diag(p.data) for p in parts]
        main = max(diags)
        big = diags.index(main)
        if big != 0:
            # keep the biggest part inside the work object so it never drops
            parts[0].data, parts[big].data = parts[big].data, parts[0].data
            diags[0], diags[big] = diags[big], diags[0]
        dropped = 0
        survivors = [work]
        for p, d in zip(parts[1:], diags[1:]):
            if d < main * LOOSE_PART_FRACTION:
                name = p.name
                me = p.data
                bpy.data.objects.remove(p)
                if me.users == 0:
                    bpy.data.meshes.remove(me)
                if name in ctx.created_objects:
                    ctx.created_objects.remove(name)
                dropped += 1
            else:
                survivors.append(p)
        if len(survivors) > 1:
            for name in _join_into(work, survivors[1:]):
                if name in ctx.created_objects:
                    ctx.created_objects.remove(name)
        return dropped


classes = ()
