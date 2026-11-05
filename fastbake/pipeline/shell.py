# Outer-shell extraction: reduce a complex object with interiors (a house with
# rooms and furniture, a vehicle with a cabin, a scan with internal debris) to
# its watertight EXTERIOR hull — the surface a ball of radius `probe` can touch
# from outside.
#
# Pipeline (pure Blender, no external dependencies):
#   1. input: the merge stage's voxel scaffold (manifold, world space)
#   2. morphological CLOSING by the probe radius r: dilate +r, re-mesh, erode -r.
#      The re-mesh between the two offsets matters: OpenVDB's mesh->level-set
#      sign flood-fill turns every region that the dilation sealed off from the
#      exterior into solid, so the erosion cannot reopen it and interior
#      surfaces disappear. Without it a signed-distance closing reopens doors.
#      Exact path: Geometry Nodes SDF grid nodes (Mesh to SDF Grid -> SDF Grid
#      Offset -> Grid to Mesh, iso 0). Fallback when those nodes do not exist:
#      Displace along normals + voxel remesh, which approximates the same
#      Minkowski offsets (verified within one voxel on box-built houses).
#   3. drop every connected component that is enclosed by another one (ray
#      parity test) and every sliver component, so only exterior hull(s) stay.
#   4. measure how much of the hull is INVENTED — face area farther than a
#      tolerance from the real source surface — and the worst deviation (about
#      half the widest sealed opening). Above the user's limit the stage
#      cancels with a clear message instead of silently baking a blob.
#
# Everything here is synchronous (seconds to tens of seconds on ~250k faces)
# and runs inside MergeStage.start(), which the contracts allow.

import math

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

from .context import StageError

GN_SDF_NODES = ("GeometryNodeMeshToSDFGrid", "GeometryNodeSDFGridOffset",
                "GeometryNodeGridToMesh")

# components smaller than this fraction of the largest one are slivers
SLIVER_FACE_FRACTION = 0.002
SLIVER_MIN_FACES = 24
# widest probe (in voxels) still closed on the exact SDF path
MAX_SDF_BAND_VOXELS = 16


def gn_available():
    """True when the exact SDF-grid closing can be built (Blender ships these
    Geometry Nodes in recent releases; older versions use the fallback)."""
    return all(hasattr(bpy.types, n) for n in GN_SDF_NODES)


# --- modifier helpers -----------------------------------------------------------

def _apply(obj, mod):
    name = mod.name
    with bpy.context.temp_override(object=obj, active_object=obj,
                                   selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier=name)


def voxel_remesh(obj, voxel):
    mod = obj.modifiers.new("FB_ShellVoxel", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = voxel
    mod.use_smooth_shade = False
    _apply(obj, mod)


def _sock(node, name):
    if name in node.inputs:
        return node.inputs[name]
    raise KeyError("%s has no input %r (has %s)" % (
        node.bl_idname, name, [s.name for s in node.inputs]))


def _sdf_offset_pass(obj, voxel, distance, band):
    """Mesh -> SDF grid -> offset by `distance` -> mesh, applied in place."""
    ng = bpy.data.node_groups.new("FB_ShellClose", "GeometryNodeTree")
    try:
        ng.interface.new_socket("Geometry", in_out="INPUT",
                                socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT",
                                socket_type="NodeSocketGeometry")
        gin = ng.nodes.new("NodeGroupInput")
        gout = ng.nodes.new("NodeGroupOutput")
        to_sdf = ng.nodes.new("GeometryNodeMeshToSDFGrid")
        off = ng.nodes.new("GeometryNodeSDFGridOffset")
        to_mesh = ng.nodes.new("GeometryNodeGridToMesh")
        _sock(to_sdf, "Voxel Size").default_value = voxel
        _sock(to_sdf, "Band Width").default_value = band
        _sock(off, "Distance").default_value = distance
        # level set: extract the zero iso-surface (the node's default 0.1 is
        # meant for fog volumes and would offset the whole hull)
        _sock(to_mesh, "Threshold").default_value = 0.0
        ng.links.new(gin.outputs[0], _sock(to_sdf, "Mesh"))
        ng.links.new(to_sdf.outputs[0], _sock(off, "Grid"))
        ng.links.new(off.outputs[0], _sock(to_mesh, "Grid"))
        ng.links.new(to_mesh.outputs[0], gout.inputs[0])
        mod = obj.modifiers.new("FB_ShellClose", "NODES")
        mod.node_group = ng
        _apply(obj, mod)
    finally:
        bpy.data.node_groups.remove(ng)


def _close_sdf(obj, voxel, radius):
    band = int(math.ceil(radius / voxel)) + 2
    _sdf_offset_pass(obj, voxel, +radius, band)   # dilate: seals openings < 2r
    _sdf_offset_pass(obj, voxel, -radius, band)   # erode back (sealed voids stay solid)


def _close_displace(obj, voxel, radius):
    for sign in (1.0, -1.0):
        mod = obj.modifiers.new("FB_ShellOffset", "DISPLACE")
        mod.mid_level = 0.0
        mod.strength = sign * radius
        _apply(obj, mod)
        voxel_remesh(obj, voxel)   # unions the offset surface, flood-fills sealed voids


def close_openings(obj, voxel, radius, log):
    """Morphological closing of `obj` in place. Returns the method used."""
    # The SDF narrow band must span the whole offset; past ~16 voxels the grid
    # work grows cubically and the exact path takes minutes, while the
    # displace approximation stays at seconds and is within a voxel anyway.
    if gn_available() and radius / voxel <= MAX_SDF_BAND_VOXELS:
        try:
            _close_sdf(obj, voxel, radius)
            return "sdf"
        except Exception as exc:  # node API drift, out-of-memory band, ...
            log("shell: SDF closing failed (%s) — using displace fallback" % exc)
            for m in list(obj.modifiers):
                if m.name.startswith("FB_Shell"):
                    obj.modifiers.remove(m)
    _close_displace(obj, voxel, radius)
    return "displace"


# --- topology helpers --------------------------------------------------------------

def _face_components(me):
    """Connected-component label per polygon (union-find over shared edges)."""
    n = len(me.polygons)
    parent = np.arange(n, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edge_faces = {}
    for p in me.polygons:
        for key in p.edge_keys:
            edge_faces.setdefault(key, []).append(p.index)
    for faces in edge_faces.values():
        root = find(faces[0])
        for f in faces[1:]:
            r = find(f)
            if r != root:
                parent[r] = root
    return np.array([find(i) for i in range(n)], dtype=np.int64)


def _remove_faces(me, mask):
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    doomed = [bm.faces[i] for i in np.nonzero(mask)[0]]
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(me)
    bm.free()
    me.update()


def drop_enclosed_and_slivers(obj, log):
    """Keep only exterior hull components. Returns (enclosed, slivers, kept)."""
    me = obj.data
    labels = _face_components(me)
    ids, counts = np.unique(labels, return_counts=True)
    if len(ids) == 1:
        return 0, 0, 1
    biggest = int(counts.max())
    sliver_cap = max(SLIVER_MIN_FACES, int(biggest * SLIVER_FACE_FRACTION))
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    polys = [tuple(p.vertices) for p in me.polygons]
    verts = [tuple(v) for v in co]

    enclosed, slivers, keep = set(), set(), set()
    for cid, cnt in zip(ids, counts):
        if int(cnt) < sliver_cap:
            slivers.add(int(cid))
            continue
        others = [polys[i] for i in np.nonzero(labels != cid)[0]]
        if not others:
            keep.add(int(cid))
            continue
        bvh = BVHTree.FromPolygons(verts, others, all_triangles=False)
        mine = np.nonzero(labels == cid)[0]
        vids = {v for i in mine for v in polys[i]}
        start = co[max(vids, key=lambda i: co[i][0])]      # this component's max-X vertex
        origin = Vector(start) + Vector((1e-4, 0.0, 0.0))
        direction = Vector((1.0, 0.0, 0.0))
        hits = 0
        for _ in range(512):
            loc, _n, _i, _d = bvh.ray_cast(origin, direction)
            if loc is None:
                break
            hits += 1
            origin = loc + direction * 1e-4
        (enclosed if hits % 2 else keep).add(int(cid))

    doomed = np.isin(labels, list(enclosed | slivers))
    if doomed.any():
        _remove_faces(me, doomed)
    log("shell: components kept %d, enclosed dropped %d, slivers dropped %d"
        % (len(keep), len(enclosed), len(slivers)))
    return len(enclosed), len(slivers), len(keep)


# --- measurement -------------------------------------------------------------------

def source_bvh(ctx, scale=1.0):
    """BVH over the evaluated source objects, in world space times `scale`
    (the merge stage works tiny inputs at an internal scale)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    to_space = Matrix.Scale(scale, 4)
    bm = bmesh.new()
    for name in ctx.src_names:
        obj = ctx.obj(name)
        if obj is None or obj.type != "MESH":
            continue
        evo = obj.evaluated_get(depsgraph)
        me = evo.to_mesh()
        try:
            tmp = bmesh.new()
            tmp.from_mesh(me)
            tmp.transform(to_space @ evo.matrix_world)
            scratch = bpy.data.meshes.new("FB_shell_scratch")
            tmp.to_mesh(scratch)
            tmp.free()
            bm.from_mesh(scratch)
            bpy.data.meshes.remove(scratch)
        finally:
            evo.to_mesh_clear()
    if not bm.faces:
        bm.free()
        return None
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree


def measure_bridging(obj, bvh, tolerance):
    """(bridged area fraction, worst deviation) of `obj` vs the source surface."""
    me = obj.data
    mw = obj.matrix_world
    total = bridged = 0.0
    worst = 0.0
    for p in me.polygons:
        _loc, _n, _i, dist = bvh.find_nearest(mw @ p.center)
        area = p.area
        total += area
        if dist is None:
            continue
        if dist > worst:
            worst = dist
        if dist > tolerance:
            bridged += area
    return (bridged / total if total > 0 else 0.0), worst


def bbox_diagonal(obj):
    pts = np.array([obj.matrix_world @ Vector(c) for c in obj.bound_box])
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


# --- entry point --------------------------------------------------------------------

def extract_outer_shell(ctx, work, voxel, probe_percent, max_bridged_percent,
                        scale=1.0):
    """Turn the merge scaffold `work` into its exterior hull in place.

    probe_percent       probe radius as % of the scaffold's bounding diagonal;
                        openings narrower than twice the radius are sealed.
    max_bridged_percent cancel when more than this % of the hull's area is
                        invented (farther than 2 voxels from the real surface).
    scale               the merge stage's internal scale (mesh units per world
                        unit); distances shown to the user are divided by it.
    Returns a stats dict (world units) and records a user notice; raises
    StageError when the hull would stretch too far.
    """
    diag = bbox_diagonal(work)
    radius = max(voxel * 2.0, diag * probe_percent / 100.0)
    faces_in = len(work.data.polygons)
    method = close_openings(work, voxel, radius, ctx.log)
    enclosed, slivers, kept = drop_enclosed_and_slivers(work, ctx.log)
    if len(work.data.polygons) == 0:
        raise StageError("Outer shell: nothing left after closing — lower the "
                         "probe radius")

    bvh = source_bvh(ctx, scale)
    bridged, worst = (0.0, 0.0)
    if bvh is not None:
        bridged, worst = measure_bridging(work, bvh, tolerance=2.0 * voxel)

    to_world = 1.0 / scale if scale else 1.0
    stats = {"method": method, "radius": radius * to_world, "faces_in": faces_in,
             "faces_out": len(work.data.polygons), "enclosed_dropped": enclosed,
             "slivers_dropped": slivers, "kept": kept,
             "bridged_fraction": bridged, "worst_deviation": worst * to_world}
    ctx.flags["outer_shell"] = stats
    ctx.log("shell: %s closing r=%.4g, %d -> %d faces, bridged %.1f%%, worst "
            "deviation %.4g" % (method, stats["radius"], faces_in,
                                stats["faces_out"], 100.0 * bridged,
                                stats["worst_deviation"]))

    if 100.0 * bridged > max_bridged_percent:
        raise StageError(
            "Outer shell stretched too far: %.1f%% of the hull would be invented "
            "surface (limit %.4g%%; widest sealed span ≈ %.3g). Lower the probe "
            "radius, raise the limit, or close the model's large openings first."
            % (100.0 * bridged, max_bridged_percent,
               2.0 * stats["worst_deviation"]))

    ctx.notice("Outer shell: sealed openings up to ≈%.3g, discarded %d interior "
               "part%s, %.1f%% of the hull is bridged surface"
               % (2.0 * stats["radius"], enclosed, "" if enclosed == 1 else "s",
                  100.0 * bridged))
    return stats


classes = ()
