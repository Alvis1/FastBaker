# UV stage: automatic seam placement + minimum-stretch unwrap + packing.
#
# The seam solver follows the Seamster recipe (Sheffer & Hart 2002) with the
# ZBrush-UV-Master ambient-occlusion twist (see research uv-unwrap.md):
#   1. terminal vertices at Gaussian-curvature extremes (angle deficit),
#   2. per-vertex AO via BVH hemisphere ray casting,
#   3. greedy Steiner tree: Dijkstra over the vertex graph with
#      length x AO x concavity edge costs connects the terminals into one
#      seam tree, so seams sneak through occluded concave regions and follow
#      quad edge loops,
#   4. MINIMUM_STRETCH (SLIM) unwrap + stretch-triggered relief seams,
#   5. average island scale, concave packing,
#   6. a hard QA gate — zero overlapping AND zero flipped faces or the result
#      is thrown away (plan §5 stage 3, §11 M3),
#   7. Smart UV Project fallback whenever anything above fails validation.
#
# Two repair passes keep the solver's own output past that gate instead of
# surrendering the low island count to the fallback: cut sectors that reach a
# full turn are extended through the vertex (see SECTOR_LIMIT), and flaps that
# still overlap after the unwrap are cut loose (see OVERLAP_ROUNDS). Thin parts
# are opened along their knife-edge folds before either (see FOLD_SEAM_ANGLE).
# When the fallback fails the gate as well, whichever of the two UV sets has
# fewer flipped + overlapping faces is kept (see UVStage._keep_better).

import heapq
import math
import time
import traceback

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_line_line_2d

from .. import compat
from .context import Stage, StageError

# --- seam cost constants ------------------------------------------------------
# Edge cost = length * ao_factor * concavity_factor [* view_penalty], with a
# loop bonus applied while walking the graph. All factors multiply a positive
# length, so Dijkstra stays valid.

# ao_factor = exp(-AO_WEIGHT * (1 - AO)), AO in [0..1] where 1 = fully open sky.
# A fully occluded edge (AO=0) costs exp(-2) ~ 0.135x a fully visible one, i.e.
# seams are ~7.4x cheaper inside cavities. Matches UV Master's "attract from
# ambient occlusion" behaviour.
AO_WEIGHT = 2.0

# concavity_factor = exp(CONCAVITY_GAIN * signed_dihedral). bmesh's
# calc_face_angle_signed() is positive for convex joins, negative for concave
# (verified on 5.1: cube edge = +pi/2). With gain 0.8: a 90-degree convex ridge
# costs 3.5x (seams avoid silhouettes), a 90-degree concave crease costs 0.28x
# (seams love creases). Flat surface = 1.0 (neutral).
CONCAVITY_GAIN = 0.8

# Multiplier applied when a path continues "straight" through a valence-4
# vertex (the edge sharing no face with the incoming edge). 0.7 < 1 makes
# seams follow the quad edge loops that AutoRemesher aligns to features.
LOOP_BONUS = 0.7

# protect_view: edges whose adjacent faces point toward the viewport camera
# cost 4x, pushing seams to the model's far side. Skipped in background mode.
VIEW_PENALTY = 4.0

# AO sampling budget: 16 rays/vertex keeps the pure-Python BVH ray loop under
# ~1s at 10k verts; above 30k verts only every k-th vertex is sampled and the
# rest inherit the mean of their sampled neighbours.
AO_RAYS = 16
AO_MAX_SAMPLED = 30000
AO_RAY_EPS = 1e-3    # ray origin offset along the normal, fraction of bbox diagonal
AO_RAY_RANGE = 0.5   # occluders beyond this fraction of the diagonal are ignored

# Terminal count scales with mesh size: sqrt(faces)/2, at least 4 (Seamster
# wants every strong curvature extremum cut, but too many terminals shred the
# surface into slivers).
TERMINALS_MIN = 4

# Wall-clock ceiling for the whole seam solve. start() runs synchronously
# inside one modal timer tick, so the UI (and ESC) is dead for its duration;
# past the budget the AO loop stops sampling (the rest inherit neighbour /
# mean values) and the Steiner loop stops attaching terminals. QA and the
# Smart UV Project fallback backstop any quality loss.
SOLVER_TIME_BUDGET = 5.0

# Each Steiner terminal connection is a Dijkstra over the full edge graph, so
# solve cost grows ~ terminals * edges (measured on 5.1.2: ~12s for 112
# terminals x 100k edges). Capping the product makes big meshes degrade to
# fewer terminals instead of blowing the time budget.
STEINER_COST_CAP = 4.0e6

# Relief loop: after unwrapping, if the 99th-percentile face stretch exceeds
# STRETCH_TRIGGER x median, cut one extra seam from the worst vertex to the
# nearest existing seam and re-unwrap; at most RELIEF_ROUNDS times.
STRETCH_TRIGGER = 4.0
RELIEF_ROUNDS = 2

UNWRAP_ITERATIONS = 50
UNWRAP_FULL_FACES = 20_000       # full iteration count up to this many faces
UNWRAP_MIN_ITERATIONS = 10       # never below this on huge results
RELIEF_TIME_BUDGET = 20.0        # seconds of extra unwraps the relief loop may spend
UNWRAP_MARGIN = 0.001
SMART_PROJECT_ANGLE = 66.0  # degrees, fallback only
PACK_MARGIN_PX = 4.0        # island margin in pixels at the target texture size

# Safety cap on tree-cotree handle loops (genus cuts); junk geometry could
# otherwise shred the mesh with hundreds of loops.
GENUS_LOOP_CAP = 8

# Thin parts of the source (fender lips, mirror housings, flanges ~1 cm thick)
# come back from a few-thousand-quad retopo as two sheets a few millimetres
# apart, joined by knife-edge folds. Left uncut, such a fold makes SLIM lay the
# second sheet over the first — and CONCAVITY_GAIN makes a convex fold the
# MOST expensive place for a seam (exp(0.8 * pi) ~ 12x), so the Steiner tree
# never cuts there by itself. Every edge whose two face normals are more than
# this far apart is therefore cut before the sector repair (which then extends
# the open fold ends through their vertices). Measured on eight low-polys of a
# 700k-triangle car (QuadriFlow and AutoRemesher, 3k-8k quads), flipped +
# overlapping faces after the overlap repair, summed over the eight:
#   no fold seams 2892 (an 8k AutoRemesher result folds globally: 2861),
#   120 deg 10, 135 deg 13, 150 deg 11, 165 deg 2734 (the same mesh again).
# The Smart UV Project fallback gets no fold seams: it already puts faces
# whose normals differ by more than twice its 66-degree limit into different
# islands (measured: its overlap counts did not change at 120, 135 or 150 deg).
FOLD_SEAM_ANGLE = math.radians(150.0)

# --- QA gate ------------------------------------------------------------------

# Faces whose |UV area| stays below this are rounding slivers, not geometry:
# they are ignored by both the flipped-face and the overlap count.
QA_AREA_EPS = 1e-9

# Verdict for UVs that are not usable at all (rather than merely imperfect):
# nothing downstream can bake them, so even the fallback failing this way
# stops the pipeline instead of writing garbage textures.
QA_NONFINITE = "UVs are not finite / carry no area"

# A cut vertex can only be flattened without its two cut sides crossing while
# the surface angle between two consecutive cuts stays under a full turn. A cut
# that ENDS at a saddle (3D angle sum > 2*pi) is therefore provably
# non-embeddable, and SLIM answers with overlapping (not flipped — no_flip
# holds) faces around the tip. Measured on the 'rock' benchmark: the one leaf
# tip with a 7.70 rad sector carried the only local overlap cluster. Cuts whose
# widest sector reaches this much of a full turn are extended THROUGH the
# vertex until every sector has room to open. 0.92 is measured, not guessed:
# the benchmark set is overlap-free at 0.92 and 0.90, while 0.95 and 1.00 leave
# the bent-limb mesh overlapping (SLIM crowds a sector well before it reaches a
# full turn) and 0.90 only adds islands.
SECTOR_LIMIT = 0.92 * 2.0 * math.pi
SECTOR_ROUNDS = 3
SECTOR_MAX_FIX = 24     # vertices repaired per round (each costs one Dijkstra)
SECTOR_TIME_BUDGET = 2.0

# SLIM's injectivity guarantee is local: two distant parts of one island can
# still land on top of each other. Each repair round cuts EVERY bad flap
# loose — one cut per cluster of overlapping or flipped faces (boundary ->
# flap -> boundary, so the island really splits) — then re-unwraps once;
# whatever is still bad after the rounds is cut loose face by face in one last
# unwrap, and the QA gate has the last word. The old rule (one flap per round,
# four rounds, hand over to the fallback above 24 overlapping faces) never
# started on a car with ~1 cm thin parts: 25-470 overlapping faces in 6-17
# clusters on 3-9k faces. The new one, with _relay_lone_faces after it, leaves
# 0 bad faces on seven of those eight layouts and 4-8 on the largest (8.5k
# faces, where the time cap ends the rounds). The count does not fall
# monotonically (157 -> 127 -> 130 -> 108 ... -> 12 on one of them), so a
# round without progress is no reason to stop; each round costs one unwrap,
# hence the wall-clock cap.
OVERLAP_ROUNDS = 8
OVERLAP_TIME_BUDGET = 12.0
OVERLAP_MAX_CLUSTERS = 64    # flaps cut per round (each costs a BFS + 2 Dijkstras)

# More than this share of the faces overlapping means the flattening is
# globally folded (seams that never took, a broken mesh): local cuts will not
# straighten it, so the stage hands over to the fallback without paying for
# unwraps.
OVERLAP_REPAIR_FRACTION = 0.25


# --- operator context helpers ---------------------------------------------------

def _view3d_override():
    """temp_override kwargs with a VIEW_3D area from the CURRENT window, or
    None (background / no viewport). Only area+region are overridden — never
    the window — so the operators keep the pipeline's scene and view layer
    (another window may show a different scene, where the result object is
    not in edit mode and every poll() fails)."""
    if bpy.app.background:
        return None
    window = bpy.context.window
    if not window:
        return None
    for area in window.screen.areas:
        if area.type == "VIEW_3D":
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            if region:
                return {"area": area, "region": region}
    return None


def _op(ov, op, **kwargs):
    if ov:
        with bpy.context.temp_override(**ov):
            return op(**kwargs)
    return op(**kwargs)


def _pack_properties():
    """Argument names bpy.ops.uv.pack_islands accepts on this build (the
    rotate/shape/margin methods arrived with the 3.6 packer)."""
    try:
        rna = bpy.ops.uv.pack_islands.get_rna_type()
        return {p.identifier for p in rna.properties}
    except Exception:
        return set()


def _view_forward():
    """World-space view direction of the current window's 3D viewport, or
    None."""
    if bpy.app.background:
        return None
    window = bpy.context.window
    if not window:
        return None
    for area in window.screen.areas:
        if area.type == "VIEW_3D":
            r3d = getattr(area.spaces.active, "region_3d", None)
            if r3d:
                return r3d.view_rotation @ Vector((0.0, 0.0, -1.0))
    return None


def _isolate(obj):
    """Object mode, only `obj` selected and active."""
    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
    for other in list(bpy.context.selected_objects):
        other.select_set(False)
    try:
        obj.hide_set(False)
    except RuntimeError:
        pass
    obj.hide_viewport = False
    try:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    except RuntimeError:
        raise StageError("UV stage: cannot select the retopo result object")


# --- seam solver ----------------------------------------------------------------

class _SeamSolver:
    """Computes seam edge indices for a mesh; keeps the cost graph around so
    relief seams can be routed later with the same metric."""

    def __init__(self, ctx, obj):
        self.ctx = ctx
        self.obj = obj
        self.adj = None        # vertex -> [(edge_index, other_vertex), ...]
        self.cost = None       # per-edge cost, python list
        self.opp = None        # (vertex, edge) -> straight-continuation edge
        self.edges = None      # (E, 2) vertex indices
        self.terminals = []
        self.deadline = 0.0    # wall-clock cutoff for the synchronous solve
        self.sector_fixes = 0  # cuts extended through a full-turn vertex
        self.fold_seams = 0    # knife-edge fold edges cut (FOLD_SEAM_ANGLE)

    def solve(self):
        ctx = self.ctx
        self.deadline = time.time() + SOLVER_TIME_BUDGET
        me = self.obj.data
        nv = len(me.vertices)
        ne = len(me.edges)
        nf = len(me.polygons)

        coords = np.empty(nv * 3, dtype=np.float64)
        me.vertices.foreach_get("co", coords)
        coords = coords.reshape(nv, 3)
        normals = np.empty(nv * 3, dtype=np.float64)
        me.vertex_normals.foreach_get("vector", normals)
        normals = normals.reshape(nv, 3)
        edges = np.empty(ne * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", edges)
        edges = edges.reshape(ne, 2)
        self.edges = edges

        # A retopo result with inf/NaN positions (seen once from QuadriFlow on
        # a debris-swarm scaffold) turns every metric below into NaN and ends
        # in NaN UVs that bake as garbage — stop with a readable message
        # instead. `not diag > 0` also catches a NaN diagonal.
        if not np.isfinite(coords).all():
            raise StageError("UV stage: the retopology result has non-finite "
                             "vertex positions — retopology failed")
        bbox = coords.max(axis=0) - coords.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        if not diag > 0.0:
            raise StageError("UV stage: result mesh is degenerate")

        self.adj = [[] for _ in range(nv)]
        for ei in range(ne):
            a = int(edges[ei, 0])
            b = int(edges[ei, 1])
            self.adj[a].append((ei, b))
            self.adj[b].append((ei, a))

        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.normal_update()
        try:
            t0 = time.time()
            deficit = self._angle_deficit(bm, nv)
            on_boundary = np.fromiter((v.is_boundary for v in bm.verts),
                                      dtype=bool, count=nv)
            self.opp = self._opposite_edges(bm)
            t1 = time.time()
            ao = self._vertex_ao(bm, coords, normals, diag)
            t2 = time.time()
            dihedral = np.empty(ne, dtype=np.float64)
            for e in bm.edges:
                dihedral[e.index] = e.calc_face_angle_signed(0.0)
        finally:
            bm.free()

        # cost assembly (all numpy)
        length = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
        edge_ao = (ao[edges[:, 0]] + ao[edges[:, 1]]) * 0.5
        cost = length * np.exp(-AO_WEIGHT * (1.0 - edge_ao))
        cost *= np.exp(CONCAVITY_GAIN * np.clip(dihedral, -math.pi, math.pi))
        view_dir = self._local_view_dir()
        if view_dir is not None:
            cost[self._view_facing_edges(me, ne, view_dir)] *= VIEW_PENALTY
        self.cost = np.maximum(cost, 1e-12).tolist()

        self.terminals = self._pick_terminals(deficit, nf)
        self.terminals += self._closed_part_terminals(deficit, on_boundary)
        t3 = time.time()
        seams = self._steiner_tree()
        seams |= self._genus_loops(me)
        folds = self._fold_edges(dihedral)
        self.fold_seams = len(folds)
        seams |= folds
        self.sector_fixes = self._split_wide_sectors(me, seams)
        ctx.log("uv: solver %d verts, %d terminals | curvature %.2fs, "
                "ao %.2fs, steiner %.2fs" % (nv, len(self.terminals),
                t1 - t0, t2 - t1, time.time() - t3))
        return seams

    # -- curvature terminals ---------------------------------------------------

    def _angle_deficit(self, bm, nv):
        # Gaussian curvature via angle deficit: 2*pi (pi on the boundary)
        # minus the sum of the incident face corner angles.
        deficit = np.full(nv, 2.0 * math.pi)
        for v in bm.verts:
            if v.is_boundary:
                deficit[v.index] = math.pi
        for f in bm.faces:
            for loop in f.loops:
                deficit[loop.vert.index] -= loop.calc_angle()
        return deficit

    def _pick_terminals(self, deficit, n_faces):
        nv = len(deficit)
        want = min(nv, max(TERMINALS_MIN,
                           int(round(math.sqrt(max(n_faces, 1)) / 2.0))))
        ne = len(self.cost)
        if ne:  # keep terminals * edges bounded (see STEINER_COST_CAP)
            want = max(TERMINALS_MIN, min(want, int(STEINER_COST_CAP // ne)))
        order = np.argsort(-np.abs(deficit))
        picked = []
        blocked = set()  # 2-ring suppression spreads terminals out
        for i in order.tolist():
            if i in blocked:
                continue
            picked.append(i)
            if len(picked) >= want:
                break
            blocked.add(i)
            ring1 = [w for _, w in self.adj[i]]
            blocked.update(ring1)
            for w in ring1:
                blocked.update(x for _, x in self.adj[w])
        return picked

    def _closed_part_terminals(self, deficit, on_boundary):
        """Extra terminals for closed parts the global pick left with < 2.

        Terminals are ranked by curvature over the whole mesh, so a smooth
        closed part next to a busy one can get no terminal at all. Then no
        seam reaches it, and the unwrap lays every one of its faces out on its
        own (measured: 527 one-face islands on a 527-face part of a car). Two
        terminals give the part one cut, which turns a sphere into a disk; the
        Steiner tree already roots a new tree in every part it cannot reach.
        Parts with a boundary flatten without a cut and are left alone.
        """
        nv = len(self.adj)
        part = np.full(nv, -1, dtype=np.int64)
        n_parts = 0
        for s in range(nv):
            if part[s] >= 0 or not self.adj[s]:
                continue
            part[s] = n_parts
            stack = [s]
            while stack:
                v = stack.pop()
                for _, w in self.adj[v]:
                    if part[w] < 0:
                        part[w] = n_parts
                        stack.append(w)
            n_parts += 1
        if n_parts < 2:
            return []
        count = np.zeros(n_parts, dtype=np.int64)
        for t in self.terminals:
            count[part[t]] += 1
        is_open = np.zeros(n_parts, dtype=bool)
        is_open[part[on_boundary & (part >= 0)]] = True
        needy = {c for c in range(n_parts) if count[c] < 2 and not is_open[c]}
        if not needy:
            return []
        taken = set(self.terminals)
        near = set(taken)  # never put a new terminal next to another one
        for t in taken:
            near.update(w for _, w in self.adj[t])
        order = np.argsort(-np.abs(deficit), kind="stable").tolist()
        extra = []
        for relaxed in (False, True):  # relaxed: parts too small to spread
            for i in order:
                c = int(part[i])
                if c not in needy or i in taken or (not relaxed and i in near):
                    continue
                extra.append(i)
                taken.add(i)
                near.add(i)
                near.update(w for _, w in self.adj[i])
                count[c] += 1
                if count[c] >= 2:
                    needy.discard(c)
                    if not needy:
                        break
            if not needy:
                break
        if extra:
            self.ctx.log("uv: +%d terminal(s) for closed part(s) the curvature "
                         "ranking missed" % len(extra))
        return extra

    def _fold_edges(self, dihedral):
        """Knife-edge folds (see FOLD_SEAM_ANGLE): edges whose face normals
        are more than FOLD_SEAM_ANGLE apart, whichever way they bend."""
        bent = np.abs(np.nan_to_num(dihedral)) > FOLD_SEAM_ANGLE
        folds = set(np.nonzero(bent)[0].tolist())
        if folds:
            self.ctx.log("uv: cut %d fold edge(s) sharper than %.0f deg"
                         % (len(folds), math.degrees(FOLD_SEAM_ANGLE)))
        return folds

    # -- ambient occlusion -------------------------------------------------------

    def _vertex_ao(self, bm, coords, normals, diag):
        nv = len(coords)
        bvh = BVHTree.FromBMesh(bm)

        if nv > AO_MAX_SAMPLED:
            stride = int(math.ceil(nv / AO_MAX_SAMPLED))
            sample_idx = np.arange(0, nv, stride)
        else:
            sample_idx = np.arange(nv)

        # cosine-weighted golden-spiral hemisphere pattern (local +Z = normal)
        k = np.arange(AO_RAYS, dtype=np.float64)
        u = (k + 0.5) / AO_RAYS
        local = np.stack((np.sqrt(u) * np.cos(k * 2.3999632297286533),
                          np.sqrt(u) * np.sin(k * 2.3999632297286533),
                          np.sqrt(1.0 - u)), axis=1)

        nrm = normals[sample_idx]
        nl = np.linalg.norm(nrm, axis=1, keepdims=True)
        nl[nl < 1e-12] = 1.0
        nrm = nrm / nl
        helper = np.where((np.abs(nrm[:, 0]) < 0.9)[:, None],
                          np.array((1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)))
        tan = np.cross(helper, nrm)
        tl = np.linalg.norm(tan, axis=1, keepdims=True)
        tl[tl < 1e-12] = 1.0
        tan /= tl
        bit = np.cross(nrm, tan)
        # per-vertex azimuth jitter breaks up banding on regular quad grids
        a = (sample_idx.astype(np.float64) * 2.3999632297286533) % (2.0 * math.pi)
        ca = np.cos(a)[:, None]
        sa = np.sin(a)[:, None]
        tan, bit = ca * tan + sa * bit, ca * bit - sa * tan

        dirs = (local[None, :, 0, None] * tan[:, None, :] +
                local[None, :, 1, None] * bit[:, None, :] +
                local[None, :, 2, None] * nrm[:, None, :])
        origins = coords[sample_idx] + nrm * (diag * AO_RAY_EPS)

        # BVHTree.ray_cast is a per-ray python call — this loop dominates the
        # solver runtime, hence the ray/vertex caps above. The deadline check
        # reserves at least half the time budget for the Steiner pass.
        max_dist = diag * AO_RAY_RANGE
        ray_cast = bvh.ray_cast
        origins_l = origins.tolist()
        dirs_l = dirs.tolist()
        n = len(sample_idx)
        ao_sampled = np.empty(n)
        ao_stop = min(self.deadline, time.time() + SOLVER_TIME_BUDGET * 0.5)
        done = n
        for i in range(n):
            if i and not i % 1024 and time.time() > ao_stop:
                self.ctx.log("uv: AO budget hit — sampled %d/%d verts"
                             % (i, n))
                done = i
                break
            origin = Vector(origins_l[i])
            hits = 0
            for d in dirs_l[i]:
                if ray_cast(origin, Vector(d), max_dist)[0] is not None:
                    hits += 1
            ao_sampled[i] = 1.0 - hits / AO_RAYS
        if done < n:
            sample_idx = sample_idx[:done]
            ao_sampled = ao_sampled[:done]

        default = float(ao_sampled.mean()) if len(ao_sampled) else 1.0
        ao = np.full(nv, default)
        ao[sample_idx] = ao_sampled
        if len(sample_idx) < nv:
            sampled = np.zeros(nv, dtype=bool)
            sampled[sample_idx] = True
            for v in range(nv):
                if sampled[v]:
                    continue
                vals = [ao[w] for _, w in self.adj[v] if sampled[w]]
                if vals:
                    ao[v] = sum(vals) / len(vals)
        return ao

    # -- view protection ---------------------------------------------------------

    def _local_view_dir(self):
        if not self.ctx.settings.get("advanced", {}).get("protect_view"):
            return None
        fwd = _view_forward()
        if fwd is None:
            return None
        m3 = self.obj.matrix_world.to_3x3()
        try:
            inv = m3.inverted()
        except ValueError:
            return None
        return np.array(inv @ fwd, dtype=np.float64)

    def _view_facing_edges(self, me, ne, view_dir):
        # An edge is "protected" when any adjacent face points at the viewer
        # (world normal opposes the view forward vector).
        nf = len(me.polygons)
        nl = len(me.loops)
        pn = np.empty(nf * 3, dtype=np.float64)
        me.polygons.foreach_get("normal", pn)
        facing = pn.reshape(nf, 3) @ view_dir < 0.0
        loop_total = np.empty(nf, dtype=np.int64)
        me.polygons.foreach_get("loop_total", loop_total)
        loop_face = np.repeat(np.arange(nf), loop_total)
        loop_edge = np.empty(nl, dtype=np.int64)
        me.loops.foreach_get("edge_index", loop_edge)
        edge_facing = np.zeros(ne, dtype=bool)
        np.logical_or.at(edge_facing, loop_edge, facing[loop_face])
        return edge_facing

    # -- quad-loop bonus table -----------------------------------------------------

    def _opposite_edges(self, bm):
        # At a valence-4 vertex the "straight continuation" of an incoming
        # edge is the one incident edge sharing no face with it.
        opp = {}
        for v in bm.verts:
            link = v.link_edges
            if len(link) != 4:
                continue
            for e in link:
                efaces = {f.index for f in e.link_faces}
                for e2 in link:
                    if e2 is e:
                        continue
                    if not any(f.index in efaces for f in e2.link_faces):
                        opp[(v.index, e.index)] = e2.index
                        break
        return opp

    # -- graph search ----------------------------------------------------------------

    def shortest_path(self, source_verts, target_verts, blocked=None):
        """Multi-source Dijkstra to the nearest target vertex.

        States are (vertex, incoming_edge) so the quad-loop bonus stays
        path-dependent yet optimal. Returns (edge_indices, vertex_indices)
        of the path (target first) or (None, None) when unreachable.
        `blocked` vertices are never entered, which is how a second cut out of
        the same source is forced to take a different route.
        """
        inf = float("inf")
        targets = target_verts if isinstance(target_verts, (set, frozenset)) \
            else set(target_verts)
        dist = {}
        pred = {}
        heap = []
        for s in source_verts:
            dist[(s, -1)] = 0.0
            heap.append((0.0, s, -1))
        heapq.heapify(heap)
        adj = self.adj
        cost = self.cost
        opp = self.opp
        end = None
        while heap:
            d, v, e_in = heapq.heappop(heap)
            if d > dist.get((v, e_in), inf):
                continue
            if v in targets:
                end = (v, e_in)
                break
            straight = opp.get((v, e_in), -1)
            for e, w in adj[v]:
                if e == e_in or (blocked is not None and w in blocked):
                    continue
                c = cost[e]
                if e == straight:
                    c *= LOOP_BONUS
                nd = d + c
                key = (w, e)
                if nd < dist.get(key, inf):
                    dist[key] = nd
                    pred[key] = (v, e_in)
                    heapq.heappush(heap, (nd, w, e))
        if end is None:
            return None, None
        path_edges = []
        path_verts = [end[0]]
        key = end
        while key[1] != -1:
            path_edges.append(key[1])
            key = pred[key]
            path_verts.append(key[0])
        return path_edges, path_verts

    def _steiner_tree(self):
        # Greedy Steiner approximation: grow one tree, always attaching the
        # terminal cheapest to reach from the current tree.
        seams = set()
        if len(self.terminals) < 2:
            return seams
        tree = {self.terminals[0]}
        remaining = set(self.terminals[1:])
        while remaining:
            if time.time() > self.deadline:
                self.ctx.log("uv: solver time budget hit — %d terminal(s) "
                             "left unconnected" % len(remaining))
                break
            path_edges, path_verts = self.shortest_path(tree, remaining)
            if path_edges is None:
                # disconnected mesh part: seed a new tree root there so its
                # terminals still get connected among themselves
                tree.add(remaining.pop())
                continue
            remaining.discard(path_verts[0])
            seams.update(path_edges)
            tree.update(path_verts)
        return seams

    # -- cut sector repair -------------------------------------------------------

    def _cut_verts(self, bm, seams):
        out = set()
        for ei in seams:
            out.add(int(self.edges[ei, 0]))
            out.add(int(self.edges[ei, 1]))
        for e in bm.edges:
            if e.is_boundary:
                out.add(e.verts[0].index)
                out.add(e.verts[1].index)
        return out

    def _sectors_at(self, vert, seams):
        """Angular sectors of `vert` between consecutive cuts.

        Each entry is (total angle, [per-face corner angles],
        [non-cut edges crossed, in fan order]) so the widest sector can be
        bisected by the edge that splits its angle in half. Empty when the
        vertex touches no cut at all.
        """
        cuts = [e for e in vert.link_edges
                if e.index in seams or e.is_boundary]
        if not cuts:
            return []
        info = {}   # face -> (edge to next vert, edge from prev vert, corner)
        for face in vert.link_faces:
            loop = next((l for l in face.loops if l.vert.index == vert.index),
                        None)
            if loop is not None:
                info[face.index] = (loop.edge, loop.link_loop_prev.edge,
                                    loop.calc_angle())
        sectors = []
        seen = set()
        for start in cuts:
            for face in start.link_faces:
                if face.index in seen or face.index not in info:
                    continue
                total = 0.0
                angles = []
                crossed = []
                cur_e, cur_f = start, face
                while True:
                    if cur_f.index in seen:
                        break
                    seen.add(cur_f.index)
                    e1, e2, angle = info[cur_f.index]
                    total += angle
                    angles.append(angle)
                    nxt = e2 if e1.index == cur_e.index else e1
                    if nxt.index in seams or nxt.is_boundary:
                        break
                    crossed.append(nxt)
                    nf = [g for g in nxt.link_faces
                          if g.index != cur_f.index and g.index in info]
                    if not nf:
                        break
                    cur_e, cur_f = nxt, nf[0]
                sectors.append((total, angles, crossed))
        return sectors

    def _split_wide_sectors(self, me, seams):
        """Extend cuts that wrap (nearly) a full turn around their vertex.

        A cut ending at a saddle leaves a boundary tip whose surface angle
        exceeds 2*pi — it cannot be laid flat without the two sides of the cut
        crossing, which is exactly the local overlap the QA gate rejects. The
        fix is the Seamster rule: cuts pass THROUGH such vertices. The
        bisecting edge plus a Dijkstra path back to the existing cut turns the
        tip into a through-cut (and splits that island in two, which is the
        point).
        """
        if not seams:
            return 0
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        deadline = time.time() + SECTOR_TIME_BUDGET
        fixed = 0
        try:
            for _ in range(SECTOR_ROUNDS):
                cut_verts = self._cut_verts(bm, seams)
                wide = [vi for vi in cut_verts
                        if any(s[0] > SECTOR_LIMIT and s[2]
                               for s in self._sectors_at(bm.verts[vi], seams))]
                if not wide:
                    break
                for vi in wide[:SECTOR_MAX_FIX]:
                    if time.time() > deadline:
                        self.ctx.log("uv: cut-sector repair budget hit — "
                                     "%d vertex(es) left" % len(wide))
                        return fixed
                    # re-check: an earlier repair in this round may have
                    # already split this vertex's sector
                    sectors = [s for s in self._sectors_at(bm.verts[vi], seams)
                               if s[0] > SECTOR_LIMIT and s[2]]
                    if not sectors:
                        continue
                    total, angles, crossed = max(sectors, key=lambda s: s[0])
                    half = total * 0.5
                    acc = 0.0
                    pick = crossed[len(crossed) // 2]
                    for i, angle in enumerate(angles):
                        acc += angle
                        if acc >= half and i < len(crossed):
                            pick = crossed[i]
                            break
                    ends = (pick.verts[0].index, pick.verts[1].index)
                    w = ends[1] if ends[0] == vi else ends[0]
                    add = {pick.index}
                    if w not in cut_verts:
                        path, _ = self.shortest_path([w], cut_verts,
                                                     blocked={vi})
                        if not path:
                            continue
                        add.update(path)
                    seams |= add
                    for ei in add:
                        cut_verts.add(int(self.edges[ei, 0]))
                        cut_verts.add(int(self.edges[ei, 1]))
                    fixed += 1
        finally:
            bm.free()
        if fixed:
            self.ctx.log("uv: extended %d cut(s) through a full-turn vertex"
                         % fixed)
        return fixed

    # -- genus / handle loops ---------------------------------------------------

    def _vertex_tree(self):
        """Cost-weighted shortest-path forest: pred[v] = (parent, edge)."""
        nv = len(self.adj)
        inf = float("inf")
        dist = [inf] * nv
        pred = [(-1, -1)] * nv
        adj = self.adj
        cost = self.cost
        for root in range(nv):
            if dist[root] < inf:
                continue
            dist[root] = 0.0
            heap = [(0.0, root)]
            while heap:
                d, v = heapq.heappop(heap)
                if d > dist[v]:
                    continue
                for e, w in adj[v]:
                    nd = d + cost[e]
                    if nd < dist[w]:
                        dist[w] = nd
                        pred[w] = (v, e)
                        heapq.heappush(heap, (nd, w))
        return pred

    def _genus_loops(self, me):
        """Tree-cotree (Erickson–Whittlesey) handle cuts.

        A seam *tree* only makes genus-0 surfaces disk-cuttable; every handle
        or tube needs a closed loop. Decompose edges into a primal spanning
        forest T (cost-weighted, so loops prefer cavities), a dual spanning
        forest T* (faces + one virtual outer node for all boundary edges),
        and leftovers X. Each x in X closes a fundamental cycle through T.
        """
        ne = len(me.edges)
        nf = len(me.polygons)
        nl = len(me.loops)
        loop_total = np.empty(nf, dtype=np.int64)
        me.polygons.foreach_get("loop_total", loop_total)
        loop_face = np.repeat(np.arange(nf), loop_total)
        loop_edge = np.empty(nl, dtype=np.int64)
        me.loops.foreach_get("edge_index", loop_edge)
        eface = [[] for _ in range(ne)]
        for li in range(nl):
            eface[loop_edge[li]].append(int(loop_face[li]))

        pred = self._vertex_tree()
        in_tree = [False] * ne
        for _, e in pred:
            if e >= 0:
                in_tree[e] = True

        outer = nf  # virtual dual node behind every boundary edge
        fedges = [[] for _ in range(nf + 1)]
        for ei in range(ne):
            faces = eface[ei]
            if len(faces) == 2:
                fedges[faces[0]].append(ei)
                fedges[faces[1]].append(ei)
            elif len(faces) == 1:
                fedges[faces[0]].append(ei)
                fedges[outer].append(ei)
        in_costar = [False] * ne
        dual_seen = [False] * (nf + 1)
        for start in range(nf + 1):
            if dual_seen[start]:
                continue
            dual_seen[start] = True
            stack = [start]
            while stack:
                f = stack.pop()
                for ei in fedges[f]:
                    if in_tree[ei] or in_costar[ei]:
                        continue
                    faces = eface[ei]
                    if len(faces) == 2:
                        g = faces[1] if faces[0] == f else faces[0]
                    else:
                        g = outer if f != outer else faces[0]
                    if dual_seen[g]:
                        continue
                    dual_seen[g] = True
                    in_costar[ei] = True
                    stack.append(g)

        loops = set()
        n_loops = 0
        for ei in range(ne):
            if in_tree[ei] or in_costar[ei] or not 1 <= len(eface[ei]) <= 2:
                continue
            if n_loops >= GENUS_LOOP_CAP:
                self.ctx.log("uv: genus loop cap reached — "
                             "extra handles left uncut")
                break
            cycle = {ei}
            for v in (int(self.edges[ei, 0]), int(self.edges[ei, 1])):
                while True:
                    parent, e = pred[v]
                    if e < 0:
                        break
                    cycle ^= {e}  # XOR: shared path above the LCA cancels out
                    v = parent
            loops |= cycle
            n_loops += 1
        if n_loops:
            self.ctx.log("uv: cut %d handle loop(s)" % n_loops)
        return loops


# --- UV measurements ---------------------------------------------------------------

def _signed_uv_areas(bm, uv_layer):
    """Shoelace UV area per face, signed, in bm.faces order.

    Signed on purpose: the sign IS the face's UV winding, which is what the
    flipped-face half of the QA gate tests (taking abs() here is how the
    orientation check used to go missing).
    """
    out = np.empty(len(bm.faces))
    for i, f in enumerate(bm.faces):
        s = 0.0
        prev = f.loops[-1][uv_layer].uv
        for loop in f.loops:
            cur = loop[uv_layer].uv
            s += prev.x * cur.y - cur.x * prev.y
            prev = cur
        out[i] = s * 0.5
    return out


def _flipped_mask(areas):
    """Faces whose UV winding disagrees with the mesh's own majority.

    Convention-free by design: mirroring every island is a convention, a
    single reversed face bakes as garbage. Sliver faces are ignored.
    """
    real = np.abs(areas) > QA_AREA_EPS
    pos = (areas > 0) & real
    neg = (areas < 0) & real
    return neg if pos.sum() >= neg.sum() else pos


def _count_flipped(areas):
    return int(_flipped_mask(areas).sum())


def _shoelace(pts):
    return 0.5 * sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1]
                     for i in range(len(pts)))


def _is_simple_polygon(pts):
    """No two non-adjacent edges of the closed 2D polygon `pts` cross."""
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # the closing edge is adjacent to the first
            if intersect_line_line_2d(a, b, pts[j], pts[(j + 1) % n]):
                return False
    return True


def _face_stretch(bm, uv_layer):
    """Per-face stretch = max(r, 1/r) of the total-normalized 3D/UV area
    ratio. Returns None when the UVs carry no usable area at all."""
    n = len(bm.faces)
    if n == 0:
        return None
    a3 = np.empty(n)
    for i, f in enumerate(bm.faces):
        a3[i] = f.calc_area()
    auv = np.abs(_signed_uv_areas(bm, uv_layer))
    if not np.all(np.isfinite(auv)):
        return None
    t3 = a3.sum()
    tuv = auv.sum()
    if t3 <= 1e-12 or tuv <= 1e-12:
        return None
    r = (a3 / t3) / np.maximum(auv / tuv, 1e-12)
    stretch = np.maximum(r, 1.0 / np.maximum(r, 1e-12))
    stretch[~np.isfinite(stretch)] = 1e6
    stretch[a3 <= 1e-12] = 1.0  # ignore degenerate 3D faces
    return stretch


def _uvs_finite_and_nonzero(bm, uv_layer):
    # by-name convenience check; the QA gate uses _areas_usable directly on
    # the area array it has already computed
    return _areas_usable(_signed_uv_areas(bm, uv_layer))


def _areas_usable(areas):
    # a NaN/inf UV coordinate poisons its face's shoelace sum, so the finite
    # test on the areas covers the coordinates as well
    if not np.all(np.isfinite(areas)):
        return False
    return float(np.abs(areas).sum()) > 1e-9


def _count_islands(bm, uv_layer):
    """Connected face components under UV continuity (loops of a shared edge
    carry identical UVs on both faces)."""
    bm.faces.ensure_lookup_table()

    def connected(e):
        if len(e.link_faces) != 2:
            return False
        uvs = {}
        fa, fb = e.link_faces
        for loop in fa.loops:
            if loop.vert in e.verts:
                uv = loop[uv_layer].uv
                uvs[loop.vert.index] = (round(uv.x, 6), round(uv.y, 6))
        for loop in fb.loops:
            if loop.vert in e.verts:
                uv = loop[uv_layer].uv
                if uvs.get(loop.vert.index) != (round(uv.x, 6), round(uv.y, 6)):
                    return False
        return True

    seen = [False] * len(bm.faces)
    islands = 0
    for f in bm.faces:
        if seen[f.index]:
            continue
        islands += 1
        seen[f.index] = True
        stack = [f]
        while stack:
            g = stack.pop()
            for e in g.edges:
                if not connected(e):
                    continue
                for h in e.link_faces:
                    if not seen[h.index]:
                        seen[h.index] = True
                        stack.append(h)
    return islands


# --- the stage -----------------------------------------------------------------------

class UVStage(Stage):
    stage_id = "uv"
    label = "Unwrapping"
    gate = True
    adjust_props = ("protect_view",)

    def start(self, ctx):
        obj = ctx.obj(ctx.result_name)
        if obj is None or obj.type != "MESH":
            raise StageError("UV stage: retopo result object is missing")
        me = obj.data
        if len(me.polygons) == 0:
            raise StageError("UV stage: result mesh has no faces")

        tex_size = float(int(ctx.settings.get("tex_size", "2048")))
        stats = {"terminals": 0, "seam_edges": 0, "islands": 0,
                 "stretch": 0.0, "relief": 0, "fallback": False,
                 "sector_fix": 0, "overlap_fix": 0, "flipped": 0,
                 "overlaps": 0, "qa": "", "fold_seams": 0, "lone_fix": 0,
                 "bad_faces": None, "kept": ""}
        ctx.flags["uv_stats"] = stats
        # SLIM's cost is superlinear in faces x iterations and it runs inside
        # ONE modal tick (no cancel, no redraw): 50 iterations freeze Blender
        # for ~36 s at 50k faces and minutes at 200k. Scale down on big results.
        n_faces = len(me.polygons)
        self._unwrap_iters = (UNWRAP_ITERATIONS if n_faces <= UNWRAP_FULL_FACES
                              else max(UNWRAP_MIN_ITERATIONS,
                                       int(UNWRAP_ITERATIONS * UNWRAP_FULL_FACES / n_faces)))

        # 1-4: seam solve on the object-mode mesh. Any solver failure routes
        # to the Smart UV Project fallback instead of failing the pipeline.
        # Object mode FIRST: seams and a UV layer written while the object is
        # in Edit Mode are discarded on the mode switch, and the seamless
        # unwrap that follows would sail through the QA gate.
        _isolate(obj)
        solver = None
        try:
            solver = _SeamSolver(ctx, obj)
            seams = solver.solve()
            seam_flags = np.zeros(len(me.edges), dtype=bool)
            if seams:
                seam_flags[list(seams)] = True
            me.edges.foreach_set("use_seam", seam_flags)
            me.update()
            stats["terminals"] = len(solver.terminals)
            stats["seam_edges"] = len(seams)
            stats["sector_fix"] = solver.sector_fixes
            stats["fold_seams"] = solver.fold_seams
        except StageError:
            raise
        except Exception:
            ctx.log(traceback.format_exc())
            ctx.notice("UV seam solver failed — used Smart UV Project instead")
            solver = None

        if not me.uv_layers:  # must happen before edit mode
            me.uv_layers.new(name="UVMap")

        ov = _view3d_override()
        try:
            bpy.ops.object.mode_set(mode="EDIT")
        except RuntimeError as exc:
            raise StageError("UV stage: cannot enter edit mode (%s)" % exc)

        try:
            solved = False
            failed = None   # the solver's rejected UV set, see _keep_better
            if solver is not None:
                try:
                    _op(ov, bpy.ops.mesh.select_all, action="SELECT")
                    self._unwrap(ov)
                    stats["relief"] = self._relief_rounds(ctx, me, ov, solver)
                    stats["overlap_fix"] = self._overlap_rounds(ctx, me, ov,
                                                                solver)
                    stats["lone_fix"] = self._relay_lone_faces(ctx, me)
                    # the overlap test leaves only its hits selected
                    _op(ov, bpy.ops.mesh.select_all, action="SELECT")
                    _op(ov, bpy.ops.uv.average_islands_scale)
                    self._pack(ctx, ov, tex_size)
                    solved = self._qa(ctx, me, ov, stats)
                    if not solved:
                        # Snapshot UVs + seams before the fallback overwrites
                        # them — but only a verdict the gate could COUNT may
                        # compete later: seams that never took, unusable UVs
                        # or an unrunnable overlap test lose outright.
                        countable = stats["bad_faces"] is not None
                        failed = (self._qa_message(stats, "Solver UVs"),
                                  dict(stats) if countable else None,
                                  self._snapshot_uvs(me) if countable else None)
                except Exception:
                    ctx.log(traceback.format_exc())
                    ctx.notice("Unwrap of solver seams failed — "
                               "used Smart UV Project instead")
            if solved:
                stats["kept"] = "solver"
            else:
                self._fallback(ctx, me, ov, tex_size)
                stats["fallback"] = True
                stats["kept"] = "fallback"
                # the fallback is the safety net, not an oracle — Smart UV
                # Project itself left 4 overlapping faces on one benchmark
                # mesh — so measure it too instead of reporting it unchecked
                if self._qa(ctx, me, ov, stats):
                    if failed is not None:
                        ctx.notice(failed[0] + " — used Smart UV Project "
                                   "instead")
                else:
                    self._keep_better(ctx, me, ov, stats, failed)
            self._final_stats(me, stats)
        finally:
            try:
                # the overlap test leaves only its hits selected — do not hand
                # that arbitrary selection back to the user
                _op(ov, bpy.ops.mesh.select_all, action="SELECT")
            except RuntimeError:
                pass
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass

        ctx.log("uv: %d islands, median stretch %.2f, QA %s%s" % (
            stats["islands"], stats["stretch"], stats["qa"] or "n/a",
            " (fallback)" if stats["fallback"] else ""))

    # -- operator wrappers ------------------------------------------------------

    def _unwrap(self, ov):
        iterations = getattr(self, "_unwrap_iters", UNWRAP_ITERATIONS)
        try:
            _op(ov, bpy.ops.uv.unwrap, method="MINIMUM_STRETCH",
                iterations=iterations, no_flip=True, margin=UNWRAP_MARGIN)
        except TypeError:
            # Blender < 4.3 has no SLIM ("Minimum Stretch") method
            _op(ov, bpy.ops.uv.unwrap, method="ANGLE_BASED", margin=UNWRAP_MARGIN)

    def _pack(self, ctx, ov, tex_size):
        # Plan §5 stage 3, verbatim: CONCAVE island shapes, CARDINAL rotation
        # (90-degree steps keep baked texels axis-aligned; free rotation
        # smears them), FRACTION margin of exactly 4 px at the target size.
        # CONCAVE is kept unconditionally: measured on 5.2.1 it costs
        # 0.1-0.9 s on real solver output (1-22 islands), and its worst case
        # (20 s) turned up on 60 ragged Smart-UV islands — the old
        # "AABB above 40 islands" switch neither matched the plan nor bounded
        # that cost (331 islands packed CONCAVE in 7 s).
        kwargs = {"shape_method": "CONCAVE", "rotate": True,
                  "margin_method": "FRACTION",
                  "margin": PACK_MARGIN_PX / tex_size}
        if "rotate_method" in _pack_properties():
            kwargs["rotate_method"] = "CARDINAL"
        t0 = time.time()
        try:
            _op(ov, bpy.ops.uv.pack_islands, **kwargs)
        except TypeError:
            # a packer older than the shape/margin methods (pre-3.6)
            _op(ov, bpy.ops.uv.pack_islands, rotate=True,
                margin=PACK_MARGIN_PX / tex_size)
        ctx.log("uv: packed in %.2fs (margin %.1f px at %d)"
                % (time.time() - t0, PACK_MARGIN_PX, int(tex_size)))

    def _fallback(self, ctx, me, ov, tex_size):
        # Must always leave valid, non-overlapping UVs. The solver's seams are
        # cleared first: smart_project honours existing seams as extra cuts
        # (measured: 6 -> 8 islands on a plain sphere), so a failed solve would
        # otherwise shred the fallback too (221 islands instead of 190 on the
        # 'rock' benchmark mesh).
        _op(ov, bpy.ops.mesh.select_all, action="SELECT")
        try:
            _op(ov, bpy.ops.mesh.mark_seam, clear=True)
        except RuntimeError as exc:
            ctx.log("uv: could not clear solver seams for the fallback (%s)"
                    % exc)
        _op(ov, bpy.ops.uv.smart_project,
            angle_limit=math.radians(SMART_PROJECT_ANGLE), island_margin=0.0)
        _op(ov, bpy.ops.uv.average_islands_scale)
        self._pack(ctx, ov, tex_size)

    # -- relief seams ---------------------------------------------------------------

    def _relief_rounds(self, ctx, me, ov, solver):
        rounds = 0
        deadline = time.time() + RELIEF_TIME_BUDGET
        for _ in range(RELIEF_ROUNDS):
            if time.time() > deadline:
                ctx.log("uv: relief rounds stopped — time budget spent")
                break
            bm = bmesh.from_edit_mesh(me)
            uv_layer = bm.loops.layers.uv.active
            if uv_layer is None:
                break
            bm.faces.ensure_lookup_table()
            stretch = _face_stretch(bm, uv_layer)
            if stretch is None:
                break
            median = float(np.median(stretch))
            worst = float(np.percentile(stretch, 99))
            if worst <= STRETCH_TRIGGER * max(median, 1e-6):
                break
            path = self._relief_path(solver, bm, int(np.argmax(stretch)))
            if not path:
                break
            bm.edges.ensure_lookup_table()
            for ei in path:
                bm.edges[ei].seam = True
            bmesh.update_edit_mesh(me)
            _op(ov, bpy.ops.mesh.select_all, action="SELECT")
            self._unwrap(ov)
            rounds += 1
            ctx.log("uv: relief seam +%d edges (p99 stretch %.1fx median)"
                    % (len(path), worst / max(median, 1e-6)))
        return rounds

    def _relief_path(self, solver, bm, worst_face):
        seam_verts = set()
        for e in bm.edges:
            if e.seam or e.is_boundary:
                seam_verts.add(e.verts[0].index)
                seam_verts.add(e.verts[1].index)
        if not seam_verts:
            return None
        source = None
        for v in bm.faces[worst_face].verts:
            if v.index not in seam_verts:
                source = v.index
                break
        if source is None:
            return None  # worst face already touches a seam everywhere
        path_edges, _ = solver.shortest_path([source], seam_verts)
        return path_edges

    # -- overlap repair --------------------------------------------------------

    def _overlap_faces(self, ctx, me, ov):
        """Faces the C-side overlap test flags, or None when it cannot run.

        compat.uv_select_sync forces sync selection + FACE select mode (5.x has
        no unsynced UV selection at all; in VERT mode the flush inflates the
        count), so the hits land in the mesh face selection. Sliver faces are
        dropped — rounding noise is not an overlap (contract decision 17).
        """
        try:
            with compat.uv_select_sync(bpy.context.scene.tool_settings):
                _op(ov, bpy.ops.mesh.select_all, action="SELECT")
                _op(ov, bpy.ops.uv.select_overlap)
                bm = bmesh.from_edit_mesh(me)
                bm.faces.ensure_lookup_table()
                uv_layer = bm.loops.layers.uv.active
                areas = (_signed_uv_areas(bm, uv_layer)
                         if uv_layer is not None else None)
                return [f.index for i, f in enumerate(bm.faces)
                        if f.select and (areas is None
                                         or abs(areas[i]) > QA_AREA_EPS)]
        except Exception as exc:
            ctx.log("uv: overlap QA unavailable (%s)" % exc)
            return None

    @staticmethod
    def _flap_source(solver, candidates, cut_verts):
        """The candidate vertex furthest (in edge hops) from any cut.

        Cutting from a vertex that already sits next to a seam only adds a
        one-edge spur — measured: it moved the overlap around (7 -> 5 -> 9
        faces) instead of removing it. The middle of the flap is where a cut
        actually relieves it. Ties go to the candidate the BFS reaches last;
        breaking them by vertex index instead left 199 instead of 91 bad
        faces over eight car low-polys, so this order is kept on purpose.
        """
        if not candidates:
            return None
        seen = set(cut_verts)
        front = list(cut_verts)
        best = None
        while front:
            nxt = []
            for v in front:
                for _, w in solver.adj[v]:
                    if w in seen:
                        continue
                    seen.add(w)
                    if w in candidates:
                        best = w      # later rings overwrite earlier ones
                    nxt.append(w)
            front = nxt
        return best

    @staticmethod
    def _face_clusters(bm, faces):
        """Overlapping faces grouped by shared vertices, biggest group first."""
        todo = set(faces)
        clusters = []
        for f0 in faces:
            if f0 not in todo:
                continue
            todo.discard(f0)
            group = [f0]
            stack = [f0]
            while stack:
                for v in bm.faces[stack.pop()].verts:
                    for g in v.link_faces:
                        if g.index in todo:
                            todo.discard(g.index)
                            group.append(g.index)
                            stack.append(g.index)
            clusters.append(group)
        clusters.sort(key=len, reverse=True)
        return clusters

    @staticmethod
    def _flap_cut(solver, bm, cluster, cut_verts):
        """Edges that cut one cluster of overlapping faces loose, or None.

        Normally two legs from the flap's middle to the existing cut. When
        every vertex of the cluster already lies on a cut (a sliver flap at an
        island border) there is no middle to start from; the interior edges
        between the cluster's faces whose both ends are on the cut then split
        it directly.
        """
        candidates = {v.index for fi in cluster for v in bm.faces[fi].verts}
        source = UVStage._flap_source(solver, candidates - cut_verts,
                                      cut_verts)
        if source is None:
            inner = set()
            members = set(cluster)
            for fi in cluster:
                for e in bm.faces[fi].edges:
                    if (not e.seam and len(e.link_faces) == 2
                            and all(f.index in members for f in e.link_faces)
                            and e.verts[0].index in cut_verts
                            and e.verts[1].index in cut_verts):
                        inner.add(e.index)
            return inner or None
        first, path_verts = solver.shortest_path([source], cut_verts)
        if not first:
            return None
        # the second leg must reach the cut somewhere else, or the two legs
        # would retrace one slit instead of separating the flap
        second, _ = solver.shortest_path([source], cut_verts,
                                         blocked=set(path_verts[:-1]))
        return set(first) | set(second or ())

    def _bad_faces(self, ctx, me, ov):
        """(faces, count) for the current layout: overlapping faces plus
        faces wound against the majority, counted the way the QA gate counts
        them (a face can be both). None when the overlap test cannot run."""
        overlapping = self._overlap_faces(ctx, me, ov)
        if overlapping is None:
            return None
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.active
        flipped = []
        if uv_layer is not None:
            flipped = np.nonzero(
                _flipped_mask(_signed_uv_areas(bm, uv_layer)))[0].tolist()
        return (sorted(set(overlapping) | set(flipped)),
                len(overlapping) + len(flipped))

    def _flap_cuts(self, solver, me, faces):
        """One flap cut per cluster of bad faces: (edges, cluster count)."""
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        cut_verts = set()
        for e in bm.edges:
            if e.seam or e.is_boundary:
                cut_verts.add(e.verts[0].index)
                cut_verts.add(e.verts[1].index)
        if not cut_verts:
            return set(), 0
        clusters = self._face_clusters(bm, faces)
        edges = set()
        for cluster in clusters[:OVERLAP_MAX_CLUSTERS]:
            legs = self._flap_cut(solver, bm, cluster, cut_verts)
            if not legs:
                continue
            edges |= legs
            for ei in legs:  # later flaps of this round may end on it
                cut_verts.add(int(solver.edges[ei, 0]))
                cut_verts.add(int(solver.edges[ei, 1]))
        return edges, len(clusters)

    @staticmethod
    def _isolating_cuts(me, faces):
        """Every not-yet-cut edge of `faces`: each becomes an island alone."""
        bm = bmesh.from_edit_mesh(me)
        bm.faces.ensure_lookup_table()
        return {e.index for fi in faces for e in bm.faces[fi].edges
                if not e.seam and len(e.link_faces) == 2}

    def _cut_and_unwrap(self, me, ov, edges):
        bm = bmesh.from_edit_mesh(me)
        bm.edges.ensure_lookup_table()
        for ei in edges:
            bm.edges[ei].seam = True
        bmesh.update_edit_mesh(me)
        _op(ov, bpy.ops.mesh.select_all, action="SELECT")
        self._unwrap(ov)

    def _overlap_rounds(self, ctx, me, ov, solver):
        """Cut flaps that still overlap after the unwrap loose, then re-unwrap.

        SLIM's no_flip guarantee is local, so two distant parts of one island
        can cover each other with every face still correctly wound. A cut that
        runs cut -> flap -> cut splits the island, which removes the overlap
        instead of hiding it. Every cluster of bad faces (overlapping or
        flipped) gets its cut in the same round, so one unwrap pays for all of
        them. What the flap rounds leave — a few faces in folded slivers — is
        then cut loose face by face (a lone face cannot overlap itself), in
        one last unwrap. A cut can also make things worse (measured: 32 -> 55
        faces in one round), so the best state seen — UVs and seams — is what
        the repair leaves behind.
        """
        rounds = 0
        deadline = time.time() + OVERLAP_TIME_BUDGET
        n_faces = len(me.polygons)
        best = None     # (count, faces, snapshot) of the best layout so far

        def measure():
            nonlocal best
            bad = self._bad_faces(ctx, me, ov)
            if bad is not None and bad[0] and (best is None or bad[1] < best[0]):
                best = (bad[1], bad[0], self._snapshot_uvs(me))
            return bad

        bad = measure()
        if bad is None or not bad[0]:
            return rounds   # clean, or unmeasurable (the QA gate says which)
        if bad[1] > OVERLAP_REPAIR_FRACTION * n_faces:
            ctx.log("uv: %d bad face(s) — too many for a local repair, "
                    "handing over to the fallback" % bad[1])
            return rounds
        while rounds < OVERLAP_ROUNDS:
            if time.time() > deadline:
                ctx.log("uv: overlap repair budget hit — %d bad face(s) left"
                        % bad[1])
                break
            edges, n_clusters = self._flap_cuts(solver, me, bad[0])
            if not edges:
                break
            self._cut_and_unwrap(me, ov, edges)
            rounds += 1
            ctx.log("uv: overlap repair — %d bad face(s) in %d cluster(s), "
                    "cut %d edge(s)" % (bad[1], n_clusters, len(edges)))
            bad = measure()
            if bad is None:
                break
            if not bad[0]:
                return rounds
        # last resort, from the best layout seen: cut its bad faces loose
        if bad is None or bad[1] > best[0]:
            self._restore_uvs(me, best[2])
        keep = best
        edges = self._isolating_cuts(me, keep[1])
        if not edges:
            return rounds
        self._cut_and_unwrap(me, ov, edges)
        rounds += 1
        ctx.log("uv: overlap repair — cut the %d remaining bad face(s) loose "
                "(%d edge(s))" % (len(keep[1]), len(edges)))
        bad = measure()
        if bad is None or bad[1] >= keep[0]:
            self._restore_uvs(me, keep[2])
            ctx.log("uv: overlap repair went back to its best state "
                    "(%d bad face(s))" % keep[0])
        return rounds

    def _relay_lone_faces(self, ctx, me):
        """Lay lone bad faces flat in their own plane; returns how many.

        A face cut loose as an island of its own (the repair's last resort)
        can still come out of SLIM as a bow-tie — flipped by the QA count.
        Measured on the car: every flipped face left after the repair (seven
        over five AutoRemesher low-polys, 1-18 % of the median face area) was
        such a lone quad, and none of them crosses itself in 3D. Its own
        planar projection, wound like the rest of the layout and scaled to its
        texel density, is exact, and packing keeps it clear of every other
        island. A face that crosses itself even in its plane is left alone.
        """
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            return 0
        bm.faces.ensure_lookup_table()
        areas = _signed_uv_areas(bm, uv_layer)
        real = np.abs(areas) > QA_AREA_EPS
        sign = (1.0 if ((areas > 0) & real).sum() >= ((areas < 0) & real).sum()
                else -1.0)
        area3 = np.array([f.calc_area() for f in bm.faces])
        density = float(np.abs(areas).sum() / max(area3.sum(), 1e-12))
        fixed = 0
        for i, f in enumerate(bm.faces):
            if not all(e.seam or e.is_boundary for e in f.edges):
                continue  # shares its island with other faces
            uv = [tuple(loop[uv_layer].uv) for loop in f.loops]
            if (areas[i] * sign > 0 or not real[i]) and _is_simple_polygon(uv):
                continue  # correctly wound and simple: nothing to fix
            f.normal_update()
            n = f.normal
            if n.length < 0.5:
                continue  # degenerate: no plane to project into
            origin = f.calc_center_median()
            t = f.verts[1].co - f.verts[0].co
            t -= n * t.dot(n)
            if t.length < 1e-12:
                continue
            t.normalize()
            b = n.cross(t)
            flat = [((loop.vert.co - origin).dot(t),
                     (loop.vert.co - origin).dot(b)) for loop in f.loops]
            flat_area = _shoelace(flat)
            if abs(flat_area) < 1e-18 or not _is_simple_polygon(flat):
                continue
            if flat_area * sign < 0:
                flat = [(-x, y) for x, y in flat]
            scale = math.sqrt(density * area3[i] / abs(flat_area)) \
                if density > 0 else 1.0
            cu = (sum(u[0] for u in uv) / len(uv), sum(u[1] for u in uv) / len(uv))
            for loop, (x, y) in zip(f.loops, flat):
                loop[uv_layer].uv = (cu[0] + x * scale, cu[1] + y * scale)
            fixed += 1
        if fixed:
            bmesh.update_edit_mesh(me)
            ctx.log("uv: laid %d lone face(s) flat in their own plane" % fixed)
        return fixed

    # -- UV set snapshots ----------------------------------------------------

    @staticmethod
    def _snapshot_uvs(me):
        """Active-layer UVs (per loop, in bmesh order) and seam flags of the
        edit mesh. Topology never changes inside the stage, so restoring in
        the same order is exact."""
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            return None
        uvs = [tuple(loop[uv_layer].uv) for f in bm.faces for loop in f.loops]
        seams = [e.seam for e in bm.edges]
        return uvs, seams

    @staticmethod
    def _restore_uvs(me, state):
        uvs, seams = state
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.active
        it = iter(uvs)
        for f in bm.faces:
            for loop in f.loops:
                loop[uv_layer].uv = next(it)
        for e, seam in zip(bm.edges, seams):
            e.seam = seam
        bmesh.update_edit_mesh(me)

    # -- QA -----------------------------------------------------------------------

    @staticmethod
    def _qa_message(stats, subject):
        """One user-facing line for a QA verdict. A check that could not run
        is reported as unverified, not as a failure — the user should know
        which of the two happened."""
        if stats.get("overlaps", 0) < 0:
            return "%s could not be verified (%s)" % (subject,
                                                      stats.get("qa") or "?")
        return "%s failed the QA gate (%s)" % (subject, stats.get("qa") or "?")

    def _qa(self, ctx, me, ov, stats):
        """Hard gate (plan §5 stage 3, §11 M3 'zero overlaps/flips').

        The solver output is kept only when it is PROVABLY clean: finite UVs
        with area, zero flipped faces, zero overlapping faces. Anything
        unproven — no UV layer, no overlap test — fails, because the Smart UV
        Project fallback is known good and a silent overlap bakes into every
        texture the pipeline then writes.
        """
        stats["overlaps"] = 0   # no stale verdict from an earlier gate run
        stats["bad_faces"] = None   # set only once both counts are known
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            stats["qa"] = "no UV layer"
            return False
        areas = _signed_uv_areas(bm, uv_layer)
        if not _areas_usable(areas):
            stats["qa"] = QA_NONFINITE
            return False
        stats["flipped"] = _count_flipped(areas)
        # A one-island-per-face layout is what an unwrap WITHOUT seams looks
        # like: no overlaps, no flips, and useless. It happens when the seams
        # never reached the mesh — route it to the fallback instead of shipping.
        n_faces = len(bm.faces)
        if n_faces >= 32:
            islands = _count_islands(bm, uv_layer)
            if islands > max(8, n_faces // 4):
                stats["qa"] = ("%d islands for %d faces — seams did not take"
                               % (islands, n_faces))
                ctx.log("uv: QA gate rejected the UVs — " + stats["qa"])
                return False
        faces = self._overlap_faces(ctx, me, ov)
        stats["overlaps"] = -1 if faces is None else len(faces)
        if faces is None:
            stats["qa"] = "overlap check unavailable"
            return False
        stats["bad_faces"] = stats["flipped"] + len(faces)
        if stats["flipped"] or faces:
            stats["qa"] = "%d flipped, %d overlapping face(s)" % (
                stats["flipped"], len(faces))
            ctx.log("uv: QA gate rejected the UVs — " + stats["qa"])
            return False
        stats["qa"] = "clean"
        return True

    def _keep_better(self, ctx, me, ov, stats, failed):
        """Both QA gates failed: keep the UV set with fewer bad faces.

        Bad faces = flipped + overlapping, as the gate counted them. The
        fallback is the safety net, not automatically the better layout —
        on a car with ~1 cm thin parts it kept 62 overlapping faces over the
        solver's 45. The solver's set (UVs and seams, snapshotted before the
        fallback ran) competes only when the gate could count it; a tie goes
        to the solver, whose seams were placed on purpose. Unusable fallback
        UVs lose to any countable solver set.
        """
        solver_msg, solver_stats, solver_uvs = failed or (None, None, None)
        fb_qa = stats.get("qa") or "?"
        fb_bad = stats.get("bad_faces")
        fb_unusable = fb_qa == QA_NONFINITE
        if solver_uvs is not None and (fb_unusable or (
                fb_bad is not None and solver_stats["bad_faces"] <= fb_bad)):
            self._restore_uvs(me, solver_uvs)
            stats.update(solver_stats)
            stats["fallback"] = False
            stats["kept"] = "solver"
            # re-measure what is now in the mesh, so the recorded verdict is
            # the restored layout's and not just the snapshot's
            self._qa(ctx, me, ov, stats)
            if stats["bad_faces"] != solver_stats["bad_faces"]:
                ctx.log("uv: restored solver UVs measure %s (snapshot said %s)"
                        % (stats["qa"], solver_stats["qa"]))
            if fb_unusable:
                why = "the fallback's UVs are unusable"
            elif solver_stats["bad_faces"] == fb_bad:
                why = "a tie goes to the solver"
            else:
                why = "%d bad face(s) instead of %d" % (
                    solver_stats["bad_faces"], fb_bad)
            ctx.notice("Solver UVs (%s) and the Smart UV Project fallback "
                       "(%s) both failed the QA gate — kept the solver UVs "
                       "(%s)" % (solver_stats["qa"], fb_qa, why))
            return
        if fb_unusable:
            # even Smart UV Project could not produce usable UVs, and nothing
            # better is on hand: baking on them would only write garbage
            raise StageError(
                "UV stage: unwrapping produced unusable UVs "
                "(not finite / no area) — the retopology result is broken")
        if solver_uvs is not None and fb_bad is not None:
            ctx.notice("Solver UVs (%s) and the Smart UV Project fallback "
                       "(%s) both failed the QA gate — kept the Smart UV "
                       "Project fallback (%d bad face(s) instead of %d)"
                       % (solver_stats["qa"], fb_qa, fb_bad,
                          solver_stats["bad_faces"]))
            return
        # nothing to compare (the solver crashed or its verdict was not a
        # count, or the fallback could not be verified): the fallback stays
        if solver_msg:
            ctx.notice(solver_msg + " — used Smart UV Project instead")
        ctx.notice(self._qa_message(stats, "The Smart UV Project fallback"))

    # -- reporting ----------------------------------------------------------------

    def _final_stats(self, me, stats):
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            return
        stretch = _face_stretch(bm, uv_layer)
        if stretch is not None:
            stats["stretch"] = float(np.median(stretch))
        stats["islands"] = _count_islands(bm, uv_layer)

    def summary(self, ctx):
        s = ctx.flags.get("uv_stats", {})
        if s.get("fallback"):
            return ("UV: Smart Project fallback — %d islands, "
                    "median stretch %.2f, QA %s" % (s.get("islands", 0),
                                                    s.get("stretch", 0.0),
                                                    s.get("qa") or "n/a"))
        folds = (" (%d on folds)" % s["fold_seams"]) if s.get("fold_seams") else ""
        extra = ""
        if s.get("relief"):
            extra += ", %d relief cut(s)" % s["relief"]
        if s.get("sector_fix") or s.get("overlap_fix"):
            extra += ", %d repair cut(s)" % (s.get("sector_fix", 0)
                                             + s.get("overlap_fix", 0))
        # the solver set can also be the one KEPT after both gates failed
        verdict = ("no overlaps or flipped faces" if s.get("qa") == "clean"
                   else "QA %s (kept over the Smart UV fallback)"
                   % (s.get("qa") or "n/a"))
        return ("UV: %d seam edges%s from %d terminals, %d islands, "
                "median stretch %.2f%s, %s"
                % (s.get("seam_edges", 0), folds, s.get("terminals", 0),
                   s.get("islands", 0), s.get("stretch", 0.0), extra, verdict))


classes = ()
