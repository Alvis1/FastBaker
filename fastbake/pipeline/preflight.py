# Preflight: pipeline context construction, scene validation and the
# apply-transforms step that precedes merging.
#
# make_context() is the entry point used by the BAKE / Lightmap operators:
# it snapshots the settings, picks the naming base + shared run counter and
# flags an unchanged prior bake (flags["already_baked"]) so the UI can ask
# before doing redundant work.

import hashlib
import math
import os

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from .. import naming, prefs
from ..compat import temp_override
from ..props import settings_to_dict
from .context import PipelineContext, Stage, StageError


# --- settings/source hash -----------------------------------------------------

def _flat_items(d, prefix=""):
    lines = []
    for key in sorted(d):
        val = d[key]
        if isinstance(val, dict):
            lines.extend(_flat_items(val, prefix + key + "."))
        else:
            lines.append("%s%s=%r" % (prefix, key, val))
    return lines


# Settings that never change what BAKE produces: review pauses are UI-only
# and the lm_* values only drive the separate Lightmap pipeline (which writes
# no manifest). Hashing them would make the already-baked guard silently stop
# firing after unrelated UI toggles.
_HASH_IGNORE = frozenset((
    "review_stages",
    "advanced.lm_size", "advanced.lm_samples",
    "advanced.lm_gamma", "advanced.lm_denoise",
))


def _source_digest(obj):
    """Fingerprint of one source object's geometry + modifier stack.

    Counts alone miss every count-preserving edit (a sculpt stroke without
    dyntopo, a Grab brush, proportional editing), so the vertex coordinates
    themselves go into the digest — plan §5 stage 0 promises that "source-mesh
    edits change the hash and re-enable processing".
    """
    me = obj.data if obj is not None and obj.type == "MESH" else None
    h = hashlib.sha1()
    if me is None:
        h.update(b"no-mesh")
        return h.hexdigest()
    h.update(("v%d:p%d" % (len(me.vertices), len(me.polygons))).encode("utf-8"))
    try:
        co = np.empty(len(me.vertices) * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        h.update(co.tobytes())
    except (AttributeError, RuntimeError, ValueError, MemoryError):
        pass  # counts-only fallback: never fail a bake over the guard
    for mod in getattr(obj, "modifiers", ()):
        h.update(("|%s:%s" % (mod.type, mod.name)).encode("utf-8"))
    return h.hexdigest()


def settings_hash(settings, src_names):
    """Stable sha1 over the settings snapshot + source identity.

    The material stage writes this into the result's "fastbake" manifest;
    make_context compares against it to detect an unchanged re-bake. Source
    identity = sorted names + per-source geometry/modifier digest, so editing
    a source invalidates the prior result. Output-irrelevant keys
    (_HASH_IGNORE) are excluded so UI-only toggles don't defeat the guard.
    """
    lines = [ln for ln in _flat_items(settings)
             if ln.split("=", 1)[0] not in _HASH_IGNORE]
    for name in sorted(src_names):
        lines.append("src:%s:%s" % (name, _source_digest(bpy.data.objects.get(name))))
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


# --- context construction -----------------------------------------------------

def _local(datablocks):
    """Only local IDs count toward name collisions (plan §6): linked-library
    data lives in its own (name, library) namespace and never collides."""
    return [d for d in datablocks if d.library is None]


def _existing_names(ctx):
    """Names already taken: blend images + materials + output dir file stems.

    Read-only: the output directory is NOT created here — merely pressing
    BAKE (or opening the re-bake dialog) must not touch the disk.
    """
    existing = {img.name for img in _local(bpy.data.images)}
    existing.update(mat.name for mat in _local(bpy.data.materials))
    outdir = ctx.output_path()
    if outdir:
        try:
            existing.update(naming.strip_ext(f) for f in os.listdir(outdir))
        except OSError:
            pass  # missing/unreachable output dir is reported by the stage
    return existing


def _naming_kwargs(ctx):
    adv = ctx.settings.get("advanced", {})
    return dict(style=adv.get("naming_style", "VERBOSE"),
                template=adv.get("naming_template"),
                material_template=adv.get("material_template"))


def _next_run(ctx, **kwargs):
    kwargs.update(_naming_kwargs(ctx))
    return naming.next_run(ctx.base, _existing_names(ctx), **kwargs)


def _count_textures(result_obj, manifest, ctx=None):
    """How many of the manifest's baked textures still exist as images."""
    try:
        maps = [str(m) for m in manifest.get("maps", ())]
        run = int(manifest.get("run", 0))
    except (AttributeError, TypeError, ValueError):
        return 0
    base = naming.base_name(result_obj.name)
    template = (ctx.settings.get("advanced", {}).get("naming_template")
                if ctx is not None else None)
    count = 0
    for key in maps:
        for style in naming.MAP_SUFFIX:
            try:
                name = naming.tex_name(base, key, run, style, template)
            except KeyError:
                continue  # map key from another version
            if bpy.data.images.get((name, None)):   # local images only
                count += 1
                break
    return count if count else len(maps)


def _find_prior_result(scene, settings, src_names):
    want = settings_hash(settings, src_names)
    for obj in scene.objects:
        if obj.library is not None:
            continue  # hard rule 10: only local IDs count
        manifest = obj.get("fastbake")
        if manifest is None:
            continue
        try:
            stored = manifest.get("settings_hash")
        except AttributeError:
            continue  # manifest is not dict-like
        if stored == want:
            return obj, manifest
    return None


def _apply_machine_defaults(settings):
    """Machine-level output root (plan §8) for scenes still on the default."""
    p = prefs.get_prefs()
    root = (getattr(p, "output_root", "") or "").strip() if p else ""
    if root and settings["advanced"].get("output_dir") in ("", "//textures/"):
        settings["advanced"]["output_dir"] = root
    return settings


def _flag_output_dir(ctx):
    """Mark an unresolvable output path so the UI can ask for a folder."""
    if ctx.output_path() is None:
        ctx.flags["needs_output_dir"] = (
            "This .blend has not been saved, so //textures/ has no location.")
    return ctx


def make_context(context):
    """Build a PipelineContext from the current selection.

    Returns (ctx, None) on success, (None, error_message) when the
    selection is unusable.
    """
    selected = list(context.selected_objects)
    sources = [o for o in selected if o.type == "MESH"]
    ignored = [o for o in selected if o.type != "MESH"]
    if not sources:
        if ignored:
            return None, "Select a mesh to bake (the selection has no mesh objects)."
        return None, "Select a mesh to bake."
    active = context.active_object
    if active is None or active.type != "MESH" or active not in sources:
        active = sources[0]

    settings = _apply_machine_defaults(settings_to_dict(context.scene.fastbake))
    ctx = PipelineContext(settings, [o.name for o in sources], active.name)
    if ignored:
        # plan §5 stage 0: non-mesh objects are ignored WITH A NOTICE
        shown = ", ".join(o.name for o in ignored[:3])
        if len(ignored) > 3:
            shown += ", ..."
        ctx.notice("Ignored %d non-mesh object(s) in the selection: %s"
                   % (len(ignored), shown))
    ctx.base = naming.base_name(active.name)
    ctx.run = _next_run(ctx)
    _flag_output_dir(ctx)

    prior = _find_prior_result(context.scene, settings, ctx.src_names)
    if prior is not None:
        result_obj, manifest = prior
        n = _count_textures(result_obj, manifest, ctx)
        ctx.flags["already_baked"] = "%s with %d texture%s already exists" % (
            result_obj.name, n, "" if n == 1 else "s")
    return ctx, None


def _manifest_run(obj):
    """The run number recorded in a FastBake result's manifest, or None."""
    manifest = obj.get("fastbake") if obj is not None else None
    if manifest is None:
        return None
    try:
        return int(manifest.get("run"))
    except (AttributeError, TypeError, ValueError):
        return None


def resolve_lightmap_run(ctx, obj):
    """Run number for a lightmap of `obj` (plan §6 table).

    The lightmap belongs to the bake it lights, so it reuses that run: a
    first-bake asset gets T_<Base>_lightmap, not _01. Only when the object
    carries no manifest (a hand-made mesh) does the counter fall back to
    scanning EXISTING LIGHTMAPS ONLY — the sibling color/normal/material
    names must not bump it, or the very first lightmap is mis-numbered.
    """
    run = _manifest_run(obj)
    if run is not None and run >= 0:
        adv = ctx.settings.get("advanced", {})
        name = naming.tex_name(ctx.base, "lightmap", run,
                               adv.get("naming_style", "VERBOSE"),
                               adv.get("naming_template"))
        if bpy.data.images.get(name) is None:
            return run
        # that lightmap already exists: this is a re-light, take a fresh number
    return _next_run(ctx, maps=("lightmap",), include_material=False)


def make_lightmap_context(context):
    """Build a PipelineContext for the Lightmap operator.

    The lightmap bakes the ACTIVE object only — exactly what the operator's
    poll and description advertise — so it must not go through make_context's
    selection-based fallback (which can silently substitute another object
    when the active one is unselected).
    """
    obj = context.active_object
    if obj is None or obj.type != "MESH":
        return None, "Make a mesh object active to bake a lightmap."
    settings = _apply_machine_defaults(settings_to_dict(context.scene.fastbake))
    ctx = PipelineContext(settings, [obj.name], obj.name)
    ctx.base = naming.base_name(obj.name)
    ctx.run = resolve_lightmap_run(ctx, obj)
    _flag_output_dir(ctx)
    return ctx, None


# --- environment ------------------------------------------------------------

def cycles_available():
    """True when the Cycles render engine add-on is registered.

    Every bake (and the lightmap) is a Cycles job; without it the failure
    surfaces as an AttributeError on scene.cycles deep inside a stage.
    """
    if hasattr(bpy.types.Scene, "cycles"):
        return True
    try:
        if "cycles" in bpy.context.preferences.addons:
            return True
    except (AttributeError, KeyError):
        pass
    try:
        items = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        return "CYCLES" in items.keys()
    except (AttributeError, KeyError, TypeError):
        return False


# --- material analysis ----------------------------------------------------------

def _analyze_material(mat):
    """Return (uses_alpha, has_metallic) for one material."""
    if not (mat.use_nodes and mat.node_tree):
        return mat.diffuse_color[3] < 1.0 - 1e-4, mat.metallic > 1e-4
    uses_alpha = False
    has_metallic = False
    for node in mat.node_tree.nodes:
        if node.type == "BSDF_TRANSPARENT":
            uses_alpha = True
        if node.type != "BSDF_PRINCIPLED":
            continue
        alpha = node.inputs.get("Alpha")
        if alpha is not None and (alpha.is_linked or
                                  alpha.default_value < 1.0 - 1e-4):
            uses_alpha = True
        metallic = node.inputs.get("Metallic")
        if metallic is not None and (metallic.is_linked or
                                     metallic.default_value > 1e-4):
            has_metallic = True
    # An explicit BLENDED render method also signals transparency intent.
    # (Legacy blend_method is NOT checked: 5.x defaults it to 'HASHED'.)
    if getattr(mat, "surface_render_method", "DITHERED") == "BLENDED":
        uses_alpha = True
    return uses_alpha, has_metallic


def _analyze_materials(sources):
    uses_alpha = False
    has_metallic = False
    seen = set()
    for obj in sources:
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            a, m = _analyze_material(mat)
            uses_alpha = uses_alpha or a
            has_metallic = has_metallic or m
    return uses_alpha, has_metallic


# --- stages ---------------------------------------------------------------------

class PreflightStage(Stage):
    stage_id = "preflight"
    label = "Checking"
    gate = False

    def start(self, ctx):
        # Every stage edits meshes through operators that need Object Mode;
        # launched from Edit/Sculpt Mode, modifier_apply would decline silently
        # (open sheets never get solidified) and the run would die at retopo.
        if bpy.context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError as exc:
                raise StageError("FastBake needs Object Mode to run (%s)" % exc)
        sources = [o for o in (ctx.obj(n) for n in ctx.src_names) if o is not None]
        if not sources:
            raise StageError("The selected meshes disappeared before the bake started.")
        if sum(len(o.data.polygons) for o in sources) == 0:
            raise StageError("The selected meshes have no faces to bake from.")
        if not cycles_available():
            raise StageError("Cycles render engine is not available")

        uses_alpha, has_metallic = _analyze_materials(sources)
        ctx.flags["source_uses_alpha"] = uses_alpha
        ctx.flags["has_metallic"] = has_metallic
        if uses_alpha and ctx.settings["roughness_to_alpha"]:
            ctx.notice("Source materials use alpha — roughness is kept as a "
                       "separate texture instead of packing into alpha.")
        if has_metallic:
            ctx.log("metallic detected: color will bake via emission")

        try:
            outdir = ctx.output_dir()
        except OSError as exc:
            raise StageError("Cannot create the output directory: %s" % exc)
        if not os.access(outdir, os.W_OK):
            raise StageError("Output directory is not writable: %s" % outdir)
        ctx.log("output dir: %s" % outdir)

        for obj in sources:
            ctx.src_visibility[obj.name] = (obj.hide_viewport, obj.hide_render)

    def summary(self, ctx):
        return "%d source mesh(es) checked" % len(ctx.src_names)


def _needs_apply(obj):
    """True when the object's local rotation or scale is not identity."""
    _loc, rot, scale = obj.matrix_basis.decompose()
    if any(abs(s - 1.0) > 1e-6 for s in scale):
        return True
    if any(abs(s - 1.0) > 1e-6 for s in obj.delta_scale):
        return True
    if any(abs(a) > 1e-6 for a in obj.delta_rotation_euler):
        return True
    angle = rot.angle
    if angle > math.pi:
        angle = 2.0 * math.pi - angle  # quaternion double cover
    return angle > 1e-6


def _transform_restore_hook(ctx, obj_name, pristine_name, rename_to,
                            rot, scale, drot, dscale):
    """Closure for ctx.undo_hooks: put back the pristine mesh datablock and
    the rotation/scale values ApplyTransformsStage baked into the vertices,
    so cancel/error rollback does not keep silently mutated sources.
    Captures names + plain tuples only (never bpy IDs)."""
    def restore():
        obj = bpy.data.objects.get(obj_name)
        pristine = bpy.data.meshes.get(pristine_name)
        if obj is not None and pristine is not None and obj.data != pristine:
            applied = obj.data
            obj.data = pristine
            if applied is not None and applied.users == 0:
                bpy.data.meshes.remove(applied)
            if rename_to:
                pristine.name = rename_to
        if rename_to:
            # the single-user backup is live mesh data now, not an orphan
            backups = ctx.flags.get("transform_backups")
            if backups and pristine_name in backups:
                backups.remove(pristine_name)
        if obj is not None:
            # Deltas FIRST: matrix_basis is the object's local matrix WITHOUT
            # the deltas, but Object.matrix_basis's setter divides the current
            # delta values out of what it is handed. Restoring the saved rot/
            # scale while the deltas are still the reset (1,1,1)/(0,0,0) ones
            # and only then writing the deltas back would apply them twice —
            # the object would come back at delta² size/orientation.
            obj.delta_rotation_euler = drot
            obj.delta_scale = dscale
            loc = obj.matrix_basis.to_translation()  # location was never applied
            obj.matrix_basis = (Matrix.Translation(loc)
                                @ Quaternion(rot).to_matrix().to_4x4()
                                @ Matrix.Diagonal(Vector(scale).to_4d()))
    return restore


class ApplyTransformsStage(Stage):
    stage_id = "transforms"
    label = "Applying transforms"
    gate = False

    def start(self, ctx):
        self._applied = 0
        sources = [o for o in (ctx.obj(n) for n in ctx.src_names)
                   if o is not None and _needs_apply(o)]
        if not sources:
            ctx.log("rotation/scale already applied on all sources")
            return

        for obj in sources:
            _loc, rot, scale = obj.matrix_basis.decompose()
            drot = tuple(obj.delta_rotation_euler)
            dscale = tuple(obj.delta_scale)
            rename_to = None
            # transform_apply refuses multi-user data; copy keeps other
            # users of the shared mesh untouched.
            if obj.data.users > 1:
                pristine = obj.data
                obj.data = obj.data.copy()
                ctx.notice("Made mesh data single-user on %s to apply its "
                           "transforms." % obj.name)
            else:
                # single-user data is mutated in place: keep a pristine
                # zero-user backup so rollback can put the geometry back
                pristine = obj.data.copy()
                rename_to = obj.data.name
                ctx.flags.setdefault("transform_backups", []).append(
                    pristine.name)
            ctx.undo_hooks.append(_transform_restore_hook(
                ctx, obj.name, pristine.name, rename_to,
                tuple(rot), tuple(scale), drot, dscale))

        if bpy.context.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except RuntimeError:
                pass
        try:
            with temp_override(selected_editable_objects=sources,
                               active_object=sources[0], object=sources[0]):
                bpy.ops.object.transform_apply(location=False, rotation=True,
                                               scale=True)
        except RuntimeError as exc:
            raise StageError("Could not apply rotation/scale: %s" % str(exc).strip())
        self._applied = len(sources)
        ctx.log("applied rotation+scale on %d object(s)" % len(sources))

    def summary(self, ctx):
        n = getattr(self, "_applied", 0)
        if not n:
            return "Transforms were already applied"
        return "Applied rotation & scale on %d object(s)" % n


class LightmapPreflightStage(Stage):
    stage_id = "preflight"
    label = "Checking"
    gate = False

    def start(self, ctx):
        obj = ctx.obj(ctx.active_name)
        if obj is None or obj.type != "MESH":
            raise StageError("Select a mesh object to bake a lightmap for.")
        if not obj.data.uv_layers:
            raise StageError("%s has no UV map — unwrap it first (or run BAKE)."
                             % obj.name)
        if len(obj.data.polygons) == 0:
            raise StageError("%s has no faces to bake a lightmap onto." % obj.name)
        if not cycles_available():
            raise StageError("Cycles render engine is not available")

        if not ctx.base:
            # ctx built without make_context: derive naming here
            ctx.base = naming.base_name(obj.name)
        # the lightmap shares the run number of the bake it lights (plan §6)
        ctx.run = resolve_lightmap_run(ctx, obj)

        try:
            outdir = ctx.output_dir()
        except OSError as exc:
            raise StageError("Cannot create the output directory: %s" % exc)
        if not os.access(outdir, os.W_OK):
            raise StageError("Output directory is not writable: %s" % outdir)

    def summary(self, ctx):
        return "Lightmap preflight OK for %s" % ctx.active_name


classes = ()
