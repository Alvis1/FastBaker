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

from .. import naming
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


def settings_hash(settings, src_names):
    """Stable sha1 over the settings snapshot + source identity.

    The material stage writes this into the result's "fastbake" manifest;
    make_context compares against it to detect an unchanged re-bake. Source
    identity = sorted names + per-source (vertex count, poly count), so
    editing a source invalidates the prior result.
    """
    lines = _flat_items(settings)
    for name in sorted(src_names):
        obj = bpy.data.objects.get(name)
        me = obj.data if obj is not None and obj.type == "MESH" else None
        lines.append("src:%s:%d:%d" % (
            name,
            len(me.vertices) if me else 0,
            len(me.polygons) if me else 0))
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


# --- context construction -----------------------------------------------------

def _existing_names(ctx):
    """Names already taken: blend images + materials + output dir file stems."""
    existing = {img.name for img in bpy.data.images}
    existing.update(mat.name for mat in bpy.data.materials)
    try:
        existing.update(naming.strip_ext(f) for f in os.listdir(ctx.output_dir()))
    except OSError:
        pass  # unreachable output dir is reported properly by the stage
    return existing


def _count_textures(result_obj, manifest):
    """How many of the manifest's baked textures still exist as images."""
    try:
        maps = [str(m) for m in manifest.get("maps", ())]
        run = int(manifest.get("run", 0))
    except (AttributeError, TypeError, ValueError):
        return 0
    base = naming.base_name(result_obj.name)
    count = 0
    for key in maps:
        for style in naming.MAP_SUFFIX:
            try:
                name = naming.tex_name(base, key, run, style)
            except KeyError:
                continue  # map key from another version
            if bpy.data.images.get(name):
                count += 1
                break
    return count if count else len(maps)


def _find_prior_result(scene, settings, src_names):
    want = settings_hash(settings, src_names)
    for obj in scene.objects:
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


def make_context(context):
    """Build a PipelineContext from the current selection.

    Returns (ctx, None) on success, (None, error_message) when the
    selection is unusable.
    """
    sources = [o for o in context.selected_objects if o.type == "MESH"]
    if not sources:
        return None, "Select a mesh to bake."
    active = context.active_object
    if active is None or active.type != "MESH" or active not in sources:
        active = sources[0]

    settings = settings_to_dict(context.scene.fastbake)
    ctx = PipelineContext(settings, [o.name for o in sources], active.name)
    ctx.base = naming.base_name(active.name)
    ctx.run = naming.next_run(ctx.base, _existing_names(ctx),
                              settings["advanced"]["naming_style"])

    prior = _find_prior_result(context.scene, settings, ctx.src_names)
    if prior is not None:
        result_obj, manifest = prior
        n = _count_textures(result_obj, manifest)
        ctx.flags["already_baked"] = "%s with %d texture%s already exists" % (
            result_obj.name, n, "" if n == 1 else "s")
    return ctx, None


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
        sources = [o for o in (ctx.obj(n) for n in ctx.src_names) if o is not None]
        if not sources:
            raise StageError("The selected meshes disappeared before the bake started.")
        if sum(len(o.data.polygons) for o in sources) == 0:
            raise StageError("The selected meshes have no faces to bake from.")

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
            # transform_apply refuses multi-user data; copy keeps other
            # users of the shared mesh untouched.
            if obj.data.users > 1:
                obj.data = obj.data.copy()
                ctx.notice("Made mesh data single-user on %s to apply its "
                           "transforms." % obj.name)

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

        if not ctx.base:
            # ctx built without make_context: derive naming here
            ctx.base = naming.base_name(obj.name)
            ctx.run = naming.next_run(ctx.base, _existing_names(ctx),
                                      ctx.settings["advanced"]["naming_style"])

        try:
            outdir = ctx.output_dir()
        except OSError as exc:
            raise StageError("Cannot create the output directory: %s" % exc)
        if not os.access(outdir, os.W_OK):
            raise StageError("Output directory is not writable: %s" % outdir)

    def summary(self, ctx):
        return "Lightmap preflight OK for %s" % ctx.active_name


classes = ()
