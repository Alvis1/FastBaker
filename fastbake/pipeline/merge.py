# Merge stage: snapshot the selected source meshes through the evaluated
# depsgraph, join the copies into one work object and turn it into a
# watertight manifold scaffold for the retopo stage.
#
# Path selection (see dev/CONTRACTS.md and the watertight-merge research):
# * single clean manifold source           -> keep the duplicate as-is
# * several clean closed shells            -> boolean Exact self-union
#   (empty-collection operand; feature-preserving)
# * anything open / non-manifold / failed  -> solidify open parts, voxel
#   remesh, manual loose-part cleanup, validate, retry at x0.7 then x1.5
#
# Small defects are repaired BEFORE that choice (MergeStage._repair_openings):
# coincident seams are welded, wire edges and small flaps on non-manifold
# edges dropped, pin holes filled. Only the connected parts that are still
# open afterwards are solidified, growing inward when the part has an outside,
# so a closed model with one stray flap is no longer inflated as a whole.
#
# The voxel size is auto-tuned (Detail preset, coupled to the quad target,
# lowered by a thin-shell estimate, floored by a face budget) and validation
# is per-component: every input part must still be present in the scaffold,
# either as surface or absorbed inside it. Tiny real-world-scale jobs are
# merged at an internal scale so the scaffold never lands inside QuadriFlow's
# 1e-4 edge epsilon; the work object carries the inverse (see
# _normalise_scale). MergeStage._build_scaffold is the single seam where the
# merge MODE is chosen.
#
# NOTE: the Manifold boolean solver refuses an empty operand collection
# ("Cannot execute, non-exact solver and empty collection", verified on
# 5.1.2) and silently leaves the mesh unchanged, so the self-union trick
# only works with the Exact solver.

import math

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .. import compat
from .context import Stage, StageError

# "Merge detail" preset -> bbox-diagonal divisor for the voxel size.
VOXEL_DIVISOR = {"LOW": 150.0, "MED": 300.0, "HIGH": 600.0}
MAX_VOXEL_DIVISOR = 1500.0   # memory clamp: never finer than diagonal/1500
SOLIDIFY_FACTOR = 2.5        # open-sheet thickness, in voxel sizes
LOOSE_PART_FRACTION = 0.02   # drop parts with bbox diagonal < 2% of the main part

# Auto-tuning inputs (plan section 5 stage 1). The Detail preset above sets the
# base voxel size; the quad coupling and the thin-shell estimate may only make
# it FINER -- a coarser scaffold loses silhouette, and the bake cannot get that
# back -- while the floors below are the memory backstop.
SCAFFOLD_FACES_PER_QUAD = 40.0   # aim for ~40x the quad target worth of faces
# A voxel remesh emits roughly one face per voxel^2 of surface; measured on
# this stage's own output (open plane, hemisphere, tube) it lands ~1.4x above
# that, so the estimate below is scaled to match reality.
VOXEL_FACE_FACTOR = 1.4
# 150k quads == 300k triangles == retopo.MAX_RETOPO_INPUT_FACES: producing a
# denser scaffold only feeds the decimator in the next stage.
MAX_SCAFFOLD_FACES = 150_000
THIN_SHELL_VOXELS = 2.5          # voxels across the thinnest wall we must keep

# Per-component survival validation (plan section 5 stage 1 validation, and
# the section 10 "thin sheets break the voxel merge path" mitigation).
SURVIVAL_SAMPLES = 12            # probe points per input component
SURVIVAL_VOXELS = 2.5            # surface distance still counted as "present"
COMPONENT_VERT_LIMIT = 400_000   # above this a copy counts as one component
MAX_TRACKED_COMPONENTS = 200     # probing is a python loop; keep it bounded

# Small-defect repair (MergeStage._repair_openings). One boundary edge used to
# make a whole copy an "open sheet": the entire model was solidified, which
# inflated it by half the shell thickness and doubled the surface area the
# voxel face budget pays for (measured on an imported car: one 2 cm flap ->
# low-poly 6 cm too fat everywhere, voxel 3.5 -> 4.9 cm). The limits are in
# sizes of a provisional voxel computed as if the input were closed.
HOLE_FILL_VOXELS = 8.0           # fill boundary loops up to this perimeter
FLAP_VOXELS = 4.0                # drop open flaps up to this side (area = side^2)
# A short loop is only a hole when the surface around it is bigger than any
# flat patch with that outline could be (isoperimetric bound, with margin);
# otherwise it is the whole outline of a small open piece such as a leaf card,
# and filling it would leave a zero-thickness sheet the voxel grid drops.
HOLE_SURFACE_RATIO = 1.5
REGION_MAX_FACES = 20_000        # face budget of one region probe
SEAM_WELD_FRACTION = 1e-6        # coincident-seam weld distance, x bbox diagonal
SPLIT_VERT_LIMIT = 3_000_000     # label up to this size to split off open parts
MAX_FLAP_PROBES = 20_000         # more non-manifold edges than this: skip flaps
# boundary_length^2 / area under this = a mostly closed part (car body with
# open wheel arches, open-bottomed scan); a flat square sheet scores 16.
MOSTLY_CLOSED_RATIO = 4.0


# --- mesh census (numpy over foreach_get, fast on multi-million tris) --------

def _diagnose(me):
    """Edge-manifoldness census of a mesh.

    boundary: edges with exactly one face; bad: wire edges or edges with 3+
    faces (also reported apart, as wire and multi); flipped: two-face edges
    whose faces wind the same way (inconsistent normals). Watertight manifold
    iff boundary == bad == flipped == 0.
    """
    n_edges = len(me.edges)
    n_loops = len(me.loops)
    result = {"boundary": 0, "bad": 0, "flipped": 0, "wire": 0, "multi": 0,
              "faces": len(me.polygons)}
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
    result["wire"] = int((count == 0).sum())
    result["multi"] = int((count > 2).sum())
    result["bad"] = result["wire"] + result["multi"]
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

# Local-space voxel size that is safely clear of that epsilon. Below it the
# scaffold's own edges land inside QuadriFlow's rejection window, so the whole
# job is scaled up instead (MergeStage._normalise_scale) and retopo bakes the
# scale back out of the result.
SAFE_LOCAL_VOXEL = QF_EDGE_EPS * 20.0


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


def _surface_area(me):
    n = len(me.polygons)
    if n == 0:
        return 0.0
    a = np.empty(n, dtype=np.float64)
    me.polygons.foreach_get("area", a)
    return float(a.sum())


def _mesh_volume(me):
    """|enclosed volume| (divergence theorem over the loop triangles).

    Only meaningful for a closed shell — callers gate on the edge census."""
    if hasattr(me, "calc_loop_triangles"):
        me.calc_loop_triangles()
    n = len(me.loop_triangles)
    if n == 0:
        return 0.0
    tri = np.empty(n * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def _signed_volume(me):
    """Signed divergence-theorem volume about the vertex centroid: positive
    when the normals point mostly outward. On an OPEN mesh it is the volume of
    the cone from the centroid, which still carries the orientation."""
    if hasattr(me, "calc_loop_triangles"):
        me.calc_loop_triangles()
    n = len(me.loop_triangles)
    if n == 0:
        return 0.0
    tri = np.empty(n * 3, dtype=np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    co -= co.mean(0)
    a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _boundary_length(me):
    """Total length of the edges that have exactly one face."""
    n_edges = len(me.edges)
    if n_edges == 0 or len(me.loops) == 0:
        return 0.0
    le = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("edge_index", le)
    count = np.bincount(le, minlength=n_edges)
    ev = np.empty(n_edges * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)[count == 1]
    if ev.shape[0] == 0:
        return 0.0
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    return float(np.linalg.norm(co[ev[:, 0]] - co[ev[:, 1]], axis=1).sum())


def _solidify_offset(me):
    """Solidify offset that keeps an open part's outer surface in place.

    A mostly closed part (open-bottomed scan, car body with open wheel
    arches) has an outside: grow its wall inward, against the outward
    normals, so the scaffold is not inflated. A sheet (plane, leaf, ribbon,
    open tube) has no inside, so it grows evenly around itself as before.
    """
    area = _surface_area(me)
    if area <= 0.0:
        return 0.0
    perim = _boundary_length(me)
    if perim * perim >= MOSTLY_CLOSED_RATIO * area:
        return 0.0
    vol = _signed_volume(me)
    # a pancake has no reliable inside either: demand a real volume
    if abs(vol) < 0.01 * area ** 1.5:
        return 0.0
    return -1.0 if vol > 0.0 else 1.0


def _vertex_labels(me, limit=COMPONENT_VERT_LIMIT):
    """Connected-component label per vertex, or None when not worth computing.

    Numpy hook-and-compress: neighbour-minimum label propagation followed by
    pointer jumping, which converges in O(log n) rounds. The edge list is
    sorted once, so a round is a gather plus a reduceat — fast enough for the
    hundreds-of-thousands-of-vertices meshes this stage sees.
    """
    nv = len(me.vertices)
    ne = len(me.edges)
    if nv == 0 or ne == 0 or nv > limit:
        return None
    ev = np.empty(ne * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    src = np.concatenate((ev[:, 0], ev[:, 1]))
    dst = np.concatenate((ev[:, 1], ev[:, 0]))
    order = np.argsort(src, kind="stable")
    src = src[order]
    dst = dst[order]
    starts = np.flatnonzero(np.concatenate(([True], src[1:] != src[:-1])))
    heads = src[starts]           # unique, so the scatter below is unambiguous
    lab = np.arange(nv, dtype=np.int64)
    for _ in range(64):
        new = lab.copy()
        new[heads] = np.minimum(lab[heads],
                                np.minimum.reduceat(lab[dst], starts))
        for _ in range(32):       # pointer jumping: collapse to the roots
            nxt = new[new]
            if np.array_equal(nxt, new):
                break
            new = nxt
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def _components(me):
    """Records for the connected components of a mesh worth tracking: bbox
    diagonal, smallest bbox extent (the thin-shell hint) and a spread sample
    of vertex positions (the survival probes).

    Fragments under LOOSE_PART_FRACTION of this mesh's largest component are
    skipped — the loose-part cleanup is allowed to drop them, and letting a
    speck set the thin-shell estimate would blow the whole voxel budget on
    debris. The per-component extents are reduced in numpy, so a mesh made of
    thousands of specks costs one sort rather than a python loop over them.
    """
    nv = len(me.vertices)
    if nv == 0:
        return []
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    labels = _vertex_labels(me)
    if labels is None:                      # too big to label: one component
        order = np.arange(nv)
        starts = np.zeros(1, dtype=np.int64)
    else:
        order = np.argsort(labels, kind="stable")
        sl = labels[order]
        starts = np.flatnonzero(np.concatenate(([True], sl[1:] != sl[:-1])))
    counts = np.diff(np.concatenate((starts, [nv])))
    pts = co[order]
    ext = (np.maximum.reduceat(pts, starts, axis=0)
           - np.minimum.reduceat(pts, starts, axis=0))
    diag = np.linalg.norm(ext, axis=1)
    keep = np.flatnonzero((counts >= 3)
                          & (diag >= diag.max() * LOOSE_PART_FRACTION))
    keep = keep[np.argsort(-diag[keep])][:MAX_TRACKED_COMPONENTS]
    out = []
    for k in keep:
        start, count = int(starts[k]), int(counts[k])
        sample = pts[start:start + count]
        if count > SURVIVAL_SAMPLES:
            step = count / float(SURVIVAL_SAMPLES)
            sample = sample[(np.arange(SURVIVAL_SAMPLES) * step).astype(np.int64)]
        else:
            sample = sample.copy()   # do not pin the whole coordinate array
        out.append({"diag": float(diag[k]),
                    "thin": float(ext[k].min()),
                    "verts": count,
                    "samples": sample})
    return out


def _lost_components(me, comps, tol):
    """Indices of `comps` that no longer have a body in `me`.

    A probe point counts as present when it is within `tol` of the scaffold
    surface OR inside it — a shell the union swallowed whole is absorbed, not
    lost, and the nearest face's normal gives the inside/outside sign
    reliably on the manifold, outward-wound meshes this runs against.

    A component needs a MAJORITY of present probes, not just one: parts of a
    kitbash overlap, so the few probes sitting inside the body it is attached
    to would otherwise vouch for a blade that the SDF erased completely.
    """
    if not comps:
        return []
    if len(me.polygons) == 0:
        return list(range(len(comps)))
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        bvh = BVHTree.FromBMesh(bm)
    finally:
        bm.free()
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(0) - tol, co.max(0) + tol
    lost = []
    for i, comp in enumerate(comps):
        samples = comp["samples"]
        alive = 0
        for p in samples:
            point = Vector((float(p[0]), float(p[1]), float(p[2])))
            loc, nor, _index, dist = bvh.find_nearest(point)
            if loc is None:
                continue
            # The normal sign test only means something inside the scaffold's
            # own bounds — a stray back-facing hit must not vouch for a point
            # that is nowhere near the mesh.
            enclosed = (bool(np.all(p >= lo) and np.all(p <= hi))
                        and (point - loc).dot(nor) < 0.0)
            if dist <= tol or enclosed:
                alive += 1
        if alive * 2 < len(samples):
            lost.append(i)
    return lost


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


# --- small-defect repair ------------------------------------------------------

def _grow_region(seed, max_area, max_faces=REGION_MAX_FACES):
    """Faces reachable from `seed` across two-face edges, explored until the
    region is exhausted or outgrows `max_area`.

    Returns (faces, status, is_open): status "small" = the whole region was
    seen and its area is within max_area, "large" = it outgrew max_area,
    "unknown" = the face budget ran out first. is_open = a boundary edge was
    met (meaningful for "small" regions only)."""
    region = {seed}
    stack = [seed]
    area = 0.0
    is_open = False
    while stack:
        f = stack.pop()
        area += f.calc_area()
        if area > max_area:
            return region, "large", is_open
        if len(region) > max_faces:
            return region, "unknown", is_open
        for e in f.edges:
            linked = e.link_faces
            if len(linked) == 1:
                is_open = True
            elif len(linked) == 2:
                g = linked[1] if linked[0] == f else linked[0]
                if g not in region:
                    region.add(g)
                    stack.append(g)
    return region, "small", is_open


def _boundary_loops(bm):
    """Closed chains of boundary edges as (edges, perimeter). Chains that do
    not close (bow-tie vertices can split them) are left out."""
    remaining = {e for e in bm.edges if e.is_boundary}
    loops = []
    while remaining:
        start = remaining.pop()
        first, v = start.verts
        edges = [start]
        perimeter = start.calc_length()
        closed = False
        while True:
            if v == first:
                closed = True
                break
            nxt = next((e for e in v.link_edges if e in remaining), None)
            if nxt is None:
                break
            remaining.discard(nxt)
            edges.append(nxt)
            perimeter += nxt.calc_length()
            v = nxt.other_vert(v)
        if closed:
            loops.append((edges, perimeter))
    return loops


def _orient_like_neighbours(faces):
    """Flip each new face whose winding disagrees with the old face across one
    of its edges (holes_fill does not promise either orientation)."""
    for f in faces:
        for loop in f.loops:
            others = [g for g in loop.edge.link_faces if g != f]
            if len(others) != 1:
                continue
            theirs = next((lp for lp in others[0].loops if lp.edge == loop.edge),
                          None)
            # consistent iff the two faces run the shared edge in opposite
            # directions, i.e. their loops on it start at different vertices
            if theirs is not None and theirs.vert == loop.vert:
                f.normal_flip()
            break


def _repair_mesh(me, hole_perimeter, flap_area, weld_dist):
    """Repair the small defects that would make a closed mesh count as open.

    In order: weld coincident boundary vertices (exporters split vertices
    along UV and normal seams), delete wire edges and loose vertices, delete
    small open flaps hanging off non-manifold edges, fill boundary loops up
    to `hole_perimeter` that are holes in a larger surface. Anything bigger
    is left for Solidify. Returns counts of what was changed."""
    stats = {"welded": 0, "wire": 0, "flaps": 0, "flap_faces": 0, "holes": 0}
    d = _diagnose(me)
    if not (d["boundary"] or d["bad"]):
        return stats
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        if d["boundary"] and weld_dist > 0.0:
            seam = [v for v in bm.verts if v.is_boundary]
            before = len(bm.verts)
            bmesh.ops.remove_doubles(bm, verts=seam, dist=weld_dist)
            stats["welded"] = before - len(bm.verts)

        wire = [e for e in bm.edges if e.is_wire]
        if wire:
            stats["wire"] = len(wire)
            bmesh.ops.delete(bm, geom=wire, context="EDGES")
        loose = [v for v in bm.verts if not v.link_edges]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")

        multi = [e for e in bm.edges if len(e.link_faces) > 2]
        if multi and len(multi) <= MAX_FLAP_PROBES:
            seen = set()
            doomed = set()
            for e in multi:
                for f in e.link_faces:
                    if f in seen:
                        continue
                    region, status, is_open = _grow_region(f, flap_area)
                    seen |= region
                    if status == "small" and is_open:
                        doomed |= region
                        stats["flaps"] += 1
            if doomed:
                stats["flap_faces"] = len(doomed)
                bmesh.ops.delete(bm, geom=list(doomed), context="FACES")

        fill = []
        for edges, perimeter in _boundary_loops(bm):
            if perimeter > hole_perimeter:
                continue
            need = HOLE_SURFACE_RATIO * perimeter * perimeter / (4.0 * math.pi)
            _region, status, _open = _grow_region(edges[0].link_faces[0], need)
            if status == "large":
                fill.extend(edges)
                stats["holes"] += 1
        if fill:
            new = bmesh.ops.holes_fill(bm, edges=fill, sides=0)["faces"]
            _orient_like_neighbours(new)
            bmesh.ops.triangulate(bm, faces=new, quad_method="BEAUTY",
                                  ngon_method="BEAUTY")

        if any(stats.values()) or loose:
            bm.to_mesh(me)
            me.update()
    finally:
        bm.free()
    return stats


def _delete_faces(me, indices):
    """Delete the faces at `indices` (and the edges/vertices only they used)."""
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.faces[int(i)] for i in indices],
                         context="FACES")
        bm.to_mesh(me)
    finally:
        bm.free()
    me.update()


def _apply_solidify(ctx, obj, thickness):
    """Give an open part thickness so the voxel SDF can represent it.

    The wall grows inward on a part that has an outside and evenly around a
    sheet (_solidify_offset), so the outer surface the bake projects onto
    stays where the source has it."""
    offset = _solidify_offset(obj.data)
    mod = obj.modifiers.new("FastBake Solidify", "SOLIDIFY")
    mod.thickness = thickness
    mod.offset = offset
    mod_name = mod.name
    try:
        with compat.temp_override(object=obj, active_object=obj,
                                  selected_objects=[obj]):
            res = bpy.ops.object.modifier_apply(modifier=mod_name)
        # modifier_apply can also decline without raising (poll failure, e.g.
        # not in Object Mode): treat a modifier that is still there as failed
        if res != {"FINISHED"} or obj.modifiers.get(mod_name) is not None:
            raise RuntimeError("modifier_apply returned %s" % (res,))
        ctx.log("merge: solidified open mesh %s (thickness %.5g, %s)"
                % (obj.name, thickness,
                   "evenly around the sheet" if offset == 0.0 else "inward"))
    except RuntimeError as exc:
        ctx.log("merge: solidify on %s failed: %s" % (obj.name, exc))
        ctx.notice("Could not thicken the open mesh %s — the merged scaffold "
                   "may have holes there" % obj.name.replace("_fbtmp", ""))
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
    adjust_props = ("merge_detail", "merge_mode", "shell_probe", "shell_max_bridged")

    _n_src = 0
    _tris = 0
    _defects = 0
    _lost = 0
    _scale = 1.0
    _path = "?"
    _notice = None

    def __init__(self):
        self._comps = []       # input components the scaffold must still hold

    def start(self, ctx):
        self._n_src = 0
        self._tris = 0
        self._defects = 0
        self._lost = 0
        self._scale = 1.0
        self._comps = []
        self._path = "?"
        # A gate Redo re-runs start() on this same instance: retract the
        # watertightness notice a previous attempt left in the report.
        if self._notice and self._notice in ctx.notices:
            ctx.notices.remove(self._notice)
        self._notice = None
        shell_note = getattr(self, "_shell_notice", None)
        if shell_note and shell_note in ctx.notices:
            ctx.notices.remove(shell_note)   # same for the outer-shell report
        self._shell_notice = None

        copies = self._snapshot_sources(ctx)
        self._n_src = len(copies)

        diag = _world_bbox_diag(copies)
        if diag <= 0.0:
            raise StageError("Nothing to merge — the selection has zero size")

        # Census + per-component bookkeeping before any thickening: the voxel
        # size is tuned to what the input actually contains, and the survival
        # check needs the input to compare the scaffold against.
        census = self._analyse(ctx, copies)
        if census["boundary"] or census["bad"]:
            # Small defects (split seams, a stray flap, wire edges, pin holes)
            # must not make a closed model count as an open sheet. They are
            # sized against a provisional voxel that assumes the input is
            # closed, the parts still open afterwards are split off, and the
            # census is taken again so area and thin-shell numbers follow
            # what will really be solidified.
            hint = self._voxel_size(ctx, diag, census, assume_closed=True,
                                    log=False)
            copies, changed = self._repair_openings(ctx, copies, census, hint,
                                                    diag)
            if changed:
                census = self._analyse(ctx, copies)
        voxel = self._voxel_size(ctx, diag, census)
        voxel, diag = self._normalise_scale(ctx, copies, census, voxel, diag)
        self._comps = self._survival_set(census)

        # Give the parts that are still open thickness so the voxel SDF can
        # represent them (normals were fixed in _analyse).
        for dup, rec in zip(copies, census["records"]):
            if rec["boundary"]:
                _apply_solidify(ctx, dup, SOLIDIFY_FACTOR * voxel)

        work = self._join(ctx, copies)
        clean = not (census["boundary"] or census["bad"] or census["flipped"])

        pristine = work.data.copy()
        try:
            self._build_scaffold(ctx, work, pristine, voxel, clean, census)
        finally:
            if pristine.users == 0:
                bpy.data.meshes.remove(pristine)

        # Final guard, the voxel/reuse counterpart of the boolean path's
        # recheck: no scaffold may leave this stage carrying edges inside
        # QuadriFlow's 1e-4 window (the per-attempt weld is clamped by the
        # voxel size, and a reused dense source never saw a weld at all).
        if _short_edge_count(work.data, eps=QF_EDGE_EPS):
            removed = weld_short_edges(work.data)
            ctx.log("merge: welded %d sub-1e-4 vertices out of the scaffold"
                    % removed)

        work.data.name = work.name
        if self._scale != 1.0:
            # Hand the scaffold back at real-world scale: the mesh data stays
            # big (both remeshers work in LOCAL space, where the 1e-4 epsilon
            # bites), the object carries the inverse, and retopo bakes that
            # matrix into the result so everything downstream is unit-scale.
            work.matrix_world = Matrix.Scale(1.0 / self._scale, 4)
            ctx.flags["merge_scale"] = self._scale
        self._tris = _tri_count(work.data)
        ctx.work_name = work.name
        ctx.log("merge: scaffold %s — %d tris, %s" % (work.name, self._tris, self._path))

    def _build_scaffold(self, ctx, work, pristine, voxel, clean, census):
        """Turn the joined copies into the scaffold.

        Single seam where a merge MODE is chosen — a new mode (outer-shell
        extraction, alpha wrap, ...) plugs in here, reads `voxel` and `work`
        and sets `self._path`; everything around it (scale normalisation,
        component survival, naming, tracking) stays mode-agnostic.
        """
        if clean and self._n_src == 1:
            self._path = "reused as-is"
            ctx.log("merge: single manifold source, duplicate reused as-is")
        else:
            done = False
            if clean:
                done = self._boolean_path(ctx, work, pristine, voxel)
            else:
                ctx.log("merge: robust path (%d boundary, %d non-manifold, "
                        "%d flipped edges in sources)"
                        % (census["boundary"], census["bad"], census["flipped"]))
            if not done:
                self._voxel_path(ctx, work, pristine, voxel)
        if ctx.settings["advanced"].get("merge_mode", "AUTO") == "OUTER_SHELL":
            self._outer_shell(ctx, work, voxel)

    def _outer_shell(self, ctx, work, voxel):
        """Merge mode OUTER_SHELL: reduce the scaffold to its exterior hull.

        The closing in shell.py starts from a manifold voxel surface, so a
        scaffold that came out of the boolean or reuse path is voxelised
        first. `voxel` and the work mesh are in the stage's internal space;
        shell.py converts every user-facing distance back with `scale`."""
        from . import shell
        adv = ctx.settings["advanced"]
        if self._path != "voxel path":
            _apply_voxel_remesh(work, voxel)
            weld_short_edges(work.data, dist=min(QF_EDGE_EPS * 2.0, voxel * 0.25))
            self._drop_loose_parts(ctx, work)
            ctx.log("merge: outer shell — voxelised the '%s' scaffold first"
                    % self._path)
        before = len(ctx.notices)
        shell.extract_outer_shell(ctx, work, voxel,
                                  float(adv.get("shell_probe", 3.0)),
                                  float(adv.get("shell_max_bridged", 15.0)),
                                  scale=self._scale)
        # remembered so a gate Redo retracts it instead of stacking a copy
        self._shell_notice = ctx.notices[-1] if len(ctx.notices) > before else None
        self._path += " + outer shell"

    def summary(self, ctx):
        noun = "mesh" if self._n_src == 1 else "meshes"
        bits = []
        if self._defects:
            bits.append("%d open edges" % self._defects)
        if self._lost:
            bits.append("%d part(s) lost" % self._lost)
        quality = ("scaffold with " + " and ".join(bits) if bits
                   else "watertight scaffold")
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

    def _analyse(self, ctx, copies):
        """Per-copy census plus everything the auto-tuner and the survival
        check need. Inconsistent normals are repaired here, before the numbers
        are taken (they change both the census and the volume estimate)."""
        records = []
        totals = {"boundary": 0, "bad": 0, "flipped": 0}
        area = 0.0
        area_closed = 0.0
        thin = None
        for dup in copies:
            d = _diagnose(dup.data)
            if d["flipped"]:
                _recalc_normals(dup.data)
                ctx.log("merge: recalculated inconsistent normals on " + dup.name)
                d = _diagnose(dup.data)
            comps = _components(dup.data)
            records.append({"obj": dup.name, "comps": comps,
                            "boundary": d["boundary"], "bad": d["bad"],
                            "flipped": d["flipped"]})
            for key in totals:
                totals[key] += d[key]
            a = _surface_area(dup.data)
            # An open part comes back from Solidify with two sides, so it
            # contributes twice its area to the scaffold the face budget has
            # to pay for. Only parts that are open AFTER the repair pass are
            # solidified (start() re-runs this census once it has repaired),
            # so a closed model with a stray flap is counted once.
            area += a * (2.0 if d["boundary"] else 1.0)
            area_closed += a
            # Thin-shell estimate, closed shells only: an open sheet has no
            # thickness until Solidify gives it one, and its flat bbox extent
            # would drive the voxel size to zero.
            if a > 0.0 and not (d["boundary"] or d["bad"]):
                cand = [c["thin"] for c in comps
                        if c["thin"] > 0.0 and c["verts"] >= 4]
                vol = _mesh_volume(dup.data)
                if vol > 0.0:
                    cand.append(2.0 * vol / a)   # mean wall thickness
                if cand:
                    thin = min(cand) if thin is None else min(thin, min(cand))
        census = dict(totals)
        census["records"] = records
        census["area"] = area
        census["area_closed"] = area_closed   # as if nothing were solidified
        census["thin"] = thin
        return census

    def _repair_openings(self, ctx, copies, census, voxel, diag):
        """Repair small defects on every copy that has any, then split the
        parts that are still open into their own objects so Solidify touches
        only them. Returns (copies, changed); `voxel` is the provisional,
        assume-closed voxel size the defect limits are measured in."""
        hole = HOLE_FILL_VOXELS * voxel
        flap = (FLAP_VOXELS * voxel) ** 2
        weld = diag * SEAM_WELD_FRACTION
        out = []
        changed = False
        for dup, rec in zip(copies, census["records"]):
            out.append(dup)
            if not (rec["boundary"] or rec["bad"]):
                continue
            stats = _repair_mesh(dup.data, hole, flap, weld)
            after = _diagnose(dup.data)
            if any(stats.values()):
                changed = True
                bits = []
                if stats["welded"]:
                    bits.append("welded %d seam vertices" % stats["welded"])
                if stats["wire"]:
                    bits.append("dropped %d wire edges" % stats["wire"])
                if stats["flaps"]:
                    bits.append("removed %d flap(s) (%d faces)"
                                % (stats["flaps"], stats["flap_faces"]))
                if stats["holes"]:
                    bits.append("filled %d hole(s)" % stats["holes"])
                ctx.log("merge: repaired %s: %s — now %d boundary, %d "
                        "non-manifold edges" % (dup.name, ", ".join(bits),
                                                after["boundary"], after["bad"]))
            if after["boundary"]:
                part = self._split_open_parts(ctx, dup)
                if part is not None:
                    out.append(part)
                    changed = True
        return out, changed

    def _split_open_parts(self, ctx, dup):
        """Move the connected parts of `dup` that still have boundary edges
        into a new object, so only they get solidified and the closed parts
        keep their exact surface. Returns the new object, or None when the
        copy is open throughout (or too big to label)."""
        me = dup.data
        labels = _vertex_labels(me, limit=SPLIT_VERT_LIMIT)
        if labels is None:
            return None
        n_edges = len(me.edges)
        le = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("edge_index", le)
        count = np.bincount(le, minlength=n_edges)
        ev = np.empty(n_edges * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", ev)
        open_labels = np.unique(labels[ev.reshape(-1, 2)[count == 1].ravel()])
        starts = np.empty(len(me.polygons), dtype=np.int64)
        me.polygons.foreach_get("loop_start", starts)
        lv = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", lv)
        open_face = np.isin(labels[lv[starts]], open_labels)
        if open_face.all() or not open_face.any():
            return None
        part_me = me.copy()
        _delete_faces(part_me, np.flatnonzero(~open_face))
        _delete_faces(me, np.flatnonzero(open_face))
        name = dup.name.replace("_fbtmp", "") + "_open_fbtmp"
        part_me.name = name
        part = bpy.data.objects.new(name, part_me)   # identity, like the copies
        bpy.context.scene.collection.objects.link(part)
        ctx.track_object(part)
        ctx.log("merge: split %d open face(s) of %s into %s, %d closed face(s) "
                "stay unthickened" % (int(open_face.sum()), dup.name, name,
                                      int((~open_face).sum())))
        return part

    def _voxel_size(self, ctx, diag, census, assume_closed=False, log=True):
        """Plan section 5 stage 1 auto-tuning: the Detail preset, coupled to
        the quad target and lowered by the thin-shell estimate, then clamped
        by the memory guard and the scaffold face budget. `assume_closed`
        counts open parts once (the provisional size the repair pass uses)."""
        adv = ctx.settings["advanced"]
        divisor = VOXEL_DIVISOR.get(adv.get("merge_detail", "MED"), 300.0)
        base = diag / divisor
        voxel = base
        area = census["area_closed" if assume_closed else "area"]
        quads = int(ctx.settings.get("quads") or 0)
        if area > 0.0 and quads > 0:
            # solve faces(voxel) = ~40x the quad target for the voxel size
            coupled = (VOXEL_FACE_FACTOR * area
                       / (SCAFFOLD_FACES_PER_QUAD * quads)) ** 0.5
            # One-sided on purpose: a small quad budget must never coarsen the
            # scaffold below the Detail preset — the bake needs the silhouette.
            voxel = min(voxel, coupled)
        thin = census["thin"]
        if thin:
            voxel = min(voxel, thin / THIN_SHELL_VOXELS)
        floor = diag / MAX_VOXEL_DIVISOR
        if area > 0.0:
            floor = max(floor, (VOXEL_FACE_FACTOR * area
                                / MAX_SCAFFOLD_FACES) ** 0.5)
        voxel = max(voxel, floor)
        if log:
            ctx.log("merge: voxel %.5g (detail %.5g, area %.4g, thin %s, "
                    "floor %.5g)" % (voxel, base, area,
                                     "%.5g" % thin if thin else "n/a", floor))
        return voxel

    def _normalise_scale(self, ctx, copies, census, voxel, diag):
        """Scale a tiny real-world-scale job up into a comfortable range.

        A 2 cm part in a metric scene gives voxel-sized scaffold edges under
        1e-4, which QuadriFlow rejects outright as "zero length edges" — and
        the weld that would remove them cannot run without collapsing the
        mesh. Scaling the copies up in place fixes the numerics at the source;
        the work object gets the inverse scale and retopo bakes it back into
        the result, so nothing downstream sees a non-unit scale.
        """
        if voxel <= 0.0 or voxel >= SAFE_LOCAL_VOXEL:
            return voxel, diag
        scale = SAFE_LOCAL_VOXEL / voxel
        matrix = Matrix.Scale(scale, 4)
        for dup in copies:
            dup.data.transform(matrix)
        for rec in census["records"]:
            for comp in rec["comps"]:
                comp["samples"] = comp["samples"] * scale
                comp["diag"] *= scale
                comp["thin"] *= scale
        census["area"] *= scale * scale
        census["area_closed"] *= scale * scale
        if census["thin"]:
            census["thin"] *= scale
        self._scale = scale
        ctx.log("merge: input is only %.4g units across — merging at %.4gx "
                "scale to stay clear of the remesher's 1e-4 edge epsilon"
                % (diag, scale))
        return voxel * scale, diag * scale

    def _survival_set(self, census):
        """Input components the scaffold must still contain.

        Fragments the loose-part cleanup is allowed to drop (under 2% of the
        largest part) are left out on purpose, as are stray vertices."""
        comps = []
        for rec in census["records"]:
            for comp in rec["comps"]:
                comp = dict(comp)
                comp["obj"] = rec["obj"]
                comps.append(comp)
        if not comps:
            return []
        main = max(c["diag"] for c in comps)
        keep = [c for c in comps
                if c["verts"] >= 3 and c["diag"] >= main * LOOSE_PART_FRACTION]
        keep.sort(key=lambda c: -c["diag"])
        return keep[:MAX_TRACKED_COMPONENTS]

    def _survival_report(self, ctx, me, voxel):
        """(lost count, message) for the scaffold currently in `me`."""
        if not self._comps:
            return 0, ""
        lost = _lost_components(me, self._comps, max(SURVIVAL_VOXELS * voxel,
                                                     QF_EDGE_EPS))
        if not lost:
            return 0, ""
        names = sorted({self._comps[i]["obj"].replace("_fbtmp", "")
                        for i in lost})
        return len(lost), "%d input part(s) missing from the scaffold (%s%s)" % (
            len(lost), ", ".join(names[:3]), ", ..." if len(names) > 3 else "")

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

    def _boolean_path(self, ctx, work, pristine, voxel):
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
            # A union may legitimately swallow an enclosed shell, but it must
            # not make a part vanish — the voxel path handles those better.
            lost, msg = self._survival_report(ctx, work.data, voxel)
            if lost:
                ctx.log("merge: self-union dropped geometry — %s — falling "
                        "back to voxel" % msg)
                _restore_mesh(work, pristine)
                return False
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
        """OpenVDB voxel remesh with a retry ladder (x1.0, x0.7, x1.5).

        An attempt is scored (lost input parts, defect edges) — the finer x0.7
        retry exists precisely to rescue a part the SDF swallowed, so the
        ladder must keep going while anything is missing, not only while the
        mesh is non-manifold (voxel output is manifold by construction)."""
        attempts = (voxel, voxel * 0.7, voxel * 1.5)
        scores = []     # (lost parts, defect edges) per attempt, None = unusable
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
            if not d["faces"]:
                scores.append(None)
                ctx.log("merge: voxel %.5g -> empty mesh" % vs)
                continue
            defects = d["boundary"] + d["bad"]
            lost, lost_msg = self._survival_report(ctx, work.data, vs)
            scores.append((lost, defects))
            ctx.log("merge: voxel %.5g -> %d faces, %s defect edges, "
                    "%d loose parts dropped%s"
                    % (vs, d["faces"], "no" if defects == 0 else defects,
                       dropped, " — " + lost_msg if lost else ""))
            if defects == 0 and lost == 0:
                self._path = "voxel path"
                self._defects = 0
                self._lost = 0
                return
        usable = [(s, i) for i, s in enumerate(scores) if s is not None]
        if not usable:
            raise StageError("Merge failed: voxel remesh produced no usable "
                             "geometry — try a higher Merge detail setting")
        best_score, best_i = min(usable)   # fewest lost parts, then fewest defects
        if best_i != current:
            _restore_mesh(work, pristine)
            _apply_voxel_remesh(work, attempts[best_i])
            weld_short_edges(work.data,
                             dist=min(QF_EDGE_EPS * 2.0, attempts[best_i] * 0.25))
            self._drop_loose_parts(ctx, work)
        self._path = "voxel path"
        self._lost, best_defects = best_score[0], best_score[1]
        self._defects = best_defects
        bits = []
        if best_defects:
            bits.append("%d problem edges" % best_defects)
        if self._lost:
            bits.append("%d input part(s) did not survive" % self._lost)
        self._notice = ("Merged scaffold is imperfect (%s) — retopology may "
                        "struggle; try a higher Merge detail setting"
                        % " and ".join(bits))
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
