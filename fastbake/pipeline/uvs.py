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
#   5. average island scale, concave packing, overlap QA,
#   6. Smart UV Project fallback whenever anything above fails validation.

import heapq
import math
import time
import traceback

import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

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
UNWRAP_MARGIN = 0.001
SMART_PROJECT_ANGLE = 66.0  # degrees, fallback only
PACK_MARGIN_PX = 4.0        # island margin in pixels at the target texture size

# CONCAVE + FRACTION packing is the quality setting, but its cost explodes
# with island count (measured on 5.1.2: 19s for 100 islands vs 0.02s for 1).
# Above this many islands pack rectangles instead — FRACTION still keeps the
# exact pixel margin.
PACK_CONCAVE_MAX_ISLANDS = 40

# Safety cap on tree-cotree handle loops (genus cuts); junk geometry could
# otherwise shred the mesh with hundreds of loops.
GENUS_LOOP_CAP = 8


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

        bbox = coords.max(axis=0) - coords.min(axis=0)
        diag = float(np.linalg.norm(bbox))
        if diag <= 0.0:
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
        t3 = time.time()
        seams = self._steiner_tree()
        seams |= self._genus_loops(me)
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

    def shortest_path(self, source_verts, target_verts):
        """Multi-source Dijkstra to the nearest target vertex.

        States are (vertex, incoming_edge) so the quad-loop bonus stays
        path-dependent yet optimal. Returns (edge_indices, vertex_indices)
        of the path (target first) or (None, None) when unreachable.
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
                if e == e_in:
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

def _face_stretch(bm, uv_layer):
    """Per-face stretch = max(r, 1/r) of the total-normalized 3D/UV area
    ratio. Returns None when the UVs carry no usable area at all."""
    n = len(bm.faces)
    if n == 0:
        return None
    a3 = np.empty(n)
    auv = np.empty(n)
    for i, f in enumerate(bm.faces):
        a3[i] = f.calc_area()
        s = 0.0
        prev = f.loops[-1][uv_layer].uv
        for loop in f.loops:
            cur = loop[uv_layer].uv
            s += prev.x * cur.y - cur.x * prev.y
            prev = cur
        auv[i] = abs(s) * 0.5
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
    total = 0.0
    for f in bm.faces:
        s = 0.0
        prev = f.loops[-1][uv_layer].uv
        for loop in f.loops:
            cur = loop[uv_layer].uv
            if not (math.isfinite(cur.x) and math.isfinite(cur.y)):
                return False
            s += prev.x * cur.y - cur.x * prev.y
            prev = cur
        total += abs(s) * 0.5
    return total > 1e-9


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
                 "stretch": 0.0, "relief": 0, "fallback": False}
        ctx.flags["uv_stats"] = stats

        # 1-4: seam solve on the object-mode mesh. Any solver failure routes
        # to the Smart UV Project fallback instead of failing the pipeline.
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
        except StageError:
            raise
        except Exception:
            ctx.log(traceback.format_exc())
            ctx.notice("UV seam solver failed — used Smart UV Project instead")
            solver = None

        if not me.uv_layers:  # must happen before edit mode
            me.uv_layers.new(name="UVMap")

        ov = _view3d_override()
        _isolate(obj)
        try:
            bpy.ops.object.mode_set(mode="EDIT")
        except RuntimeError as exc:
            raise StageError("UV stage: cannot enter edit mode (%s)" % exc)

        try:
            solved = False
            if solver is not None:
                try:
                    _op(ov, bpy.ops.mesh.select_all, action="SELECT")
                    self._unwrap(ov)
                    stats["relief"] = self._relief_rounds(ctx, me, ov, solver)
                    _op(ov, bpy.ops.uv.average_islands_scale)
                    self._pack(ov, me, tex_size)
                    solved = self._qa(ctx, me, ov)
                    if not solved:
                        ctx.notice("Solver UVs failed the overlap check — "
                                   "used Smart UV Project instead")
                except Exception:
                    ctx.log(traceback.format_exc())
                    ctx.notice("Unwrap of solver seams failed — "
                               "used Smart UV Project instead")
            if not solved:
                self._fallback(ctx, me, ov, tex_size)
                stats["fallback"] = True
            self._final_stats(me, stats)
        finally:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass

        ctx.log("uv: %d islands, median stretch %.2f%s" % (
            stats["islands"], stats["stretch"],
            " (fallback)" if stats["fallback"] else ""))

    # -- operator wrappers ------------------------------------------------------

    def _unwrap(self, ov):
        try:
            _op(ov, bpy.ops.uv.unwrap, method="MINIMUM_STRETCH",
                iterations=UNWRAP_ITERATIONS, no_flip=True, margin=UNWRAP_MARGIN)
        except TypeError:
            # Blender < 4.3 has no SLIM ("Minimum Stretch") method
            _op(ov, bpy.ops.uv.unwrap, method="ANGLE_BASED", margin=UNWRAP_MARGIN)

    def _pack(self, ov, me, tex_size):
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.active
        n_islands = _count_islands(bm, uv_layer) if uv_layer else 0
        shape = "CONCAVE" if n_islands <= PACK_CONCAVE_MAX_ISLANDS else "AABB"
        try:
            _op(ov, bpy.ops.uv.pack_islands, shape_method=shape,
                rotate=True, margin_method="FRACTION",
                margin=PACK_MARGIN_PX / tex_size)
        except TypeError:
            _op(ov, bpy.ops.uv.pack_islands, rotate=True,
                margin=PACK_MARGIN_PX / tex_size)

    def _fallback(self, ctx, me, ov, tex_size):
        # Must always leave valid, non-overlapping UVs. Existing seams are
        # honored by smart_project as extra cuts, which is harmless.
        _op(ov, bpy.ops.mesh.select_all, action="SELECT")
        _op(ov, bpy.ops.uv.smart_project,
            angle_limit=math.radians(SMART_PROJECT_ANGLE), island_margin=0.0)
        _op(ov, bpy.ops.uv.average_islands_scale)
        self._pack(ov, me, tex_size)

    # -- relief seams ---------------------------------------------------------------

    def _relief_rounds(self, ctx, me, ov, solver):
        rounds = 0
        for _ in range(RELIEF_ROUNDS):
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

    # -- QA -----------------------------------------------------------------------

    def _qa(self, ctx, me, ov):
        bm = bmesh.from_edit_mesh(me)
        uv_layer = bm.loops.layers.uv.active
        if uv_layer is None:
            return False
        if not _uvs_finite_and_nonzero(bm, uv_layer):
            return False

        # Overlap check via the C-side operator. With UV sync selection on
        # (the only mode in 5.x — BMLoopUV.select is gone) the result lands
        # in the mesh face selection.
        overlaps = None
        tool = bpy.context.scene.tool_settings
        old_sync = getattr(tool, "use_uv_select_sync", True)
        old_mode = tuple(tool.mesh_select_mode)
        try:
            try:
                tool.use_uv_select_sync = True
                # FACE select mode is required for an exact count: in VERT
                # mode (the factory default) the operator's selection flush
                # promotes any face whose verts all touch genuinely
                # overlapping faces, inflating the count past the tolerance
                # below and triggering spurious fallbacks.
                tool.mesh_select_mode = (False, False, True)
            except Exception:
                pass
            _op(ov, bpy.ops.uv.select_overlap)
            bm = bmesh.from_edit_mesh(me)
            overlaps = sum(1 for f in bm.faces if f.select)
        except Exception as exc:
            ctx.log("uv: overlap QA unavailable (%s)" % exc)
        finally:
            try:
                tool.use_uv_select_sync = old_sync
                tool.mesh_select_mode = old_mode
            except Exception:
                pass
        if overlaps is None:
            return True  # cannot verify — accept the solver output
        if overlaps:
            ctx.log("uv: %d overlapping faces detected" % overlaps)
        # tolerate rounding-level slivers, reject anything real
        return overlaps <= max(2, len(bm.faces) // 500)

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
                    "median stretch %.2f" % (s.get("islands", 0),
                                             s.get("stretch", 0.0)))
        extra = ", %d relief cut(s)" % s["relief"] if s.get("relief") else ""
        return ("UV: %d seam edges from %d terminals, %d islands, "
                "median stretch %.2f%s" % (s.get("seam_edges", 0),
                                           s.get("terminals", 0),
                                           s.get("islands", 0),
                                           s.get("stretch", 0.0), extra))


classes = ()
