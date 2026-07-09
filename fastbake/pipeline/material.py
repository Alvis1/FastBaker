# Material stage: build the final Principled material from the saved textures,
# write the provenance manifest, stash the originals in FastBake_Sources,
# remove the scaffold object and hand the result to the user.

from datetime import datetime

import bpy

from .. import naming
from .context import Stage, StageError

SOURCES_COLLECTION = "FastBake_Sources"


# The manifest hash MUST match the staleness check in preflight — single
# source of truth lives there.
from .preflight import settings_hash as _settings_hash_fn


def _settings_hash(settings, src_names):
    return _settings_hash_fn(settings, src_names)


class MaterialStage(Stage):
    stage_id = "material"
    label = "Finishing"
    gate = False

    def start(self, ctx):
        result = ctx.obj(ctx.result_name)
        if result is None or result.type != "MESH":
            raise StageError("Result mesh is missing — cannot assign material")

        mat = self._build_material(ctx)
        ctx.track_material(mat)
        ctx.material_name = mat.name

        # replace the temporary bake material(s) on the result
        old_names = [m.name for m in result.data.materials if m]
        result.data.materials.clear()
        result.data.materials.append(mat)
        for name in old_names:
            if name == mat.name or name not in ctx.created_materials:
                continue
            old = bpy.data.materials.get(name)
            if old and old.users == 0:
                bpy.data.materials.remove(old)

        self._write_provenance(ctx, result)
        self._stash_sources(ctx)
        self._remove_work_object(ctx)
        self._select_result(result)

    # -- helpers ---------------------------------------------------------

    def _build_material(self, ctx):
        mat = bpy.data.materials.new(naming.mat_name(ctx.base, ctx.run))
        mat.use_nodes = True
        tree = mat.node_tree

        bsdf = None
        for node in tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break
        if bsdf is None:
            bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
            out = None
            for node in tree.nodes:
                if node.type == "OUTPUT_MATERIAL":
                    out = node
                    break
            if out is None:
                out = tree.nodes.new("ShaderNodeOutputMaterial")
            tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

        packed = bool(ctx.flags.get("roughness_packed"))

        color = bpy.data.images.get(ctx.images.get("color", ""))
        if color:
            node = tree.nodes.new("ShaderNodeTexImage")
            node.image = color
            node.label = "Color"
            node.location = (-460, 300)
            tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
            if packed:
                # roughness lives in the color alpha (CHANNEL_PACKED)
                tree.links.new(node.outputs["Alpha"], bsdf.inputs["Roughness"])

        if not packed:
            rough = bpy.data.images.get(ctx.images.get("roughness", ""))
            if rough:
                node = tree.nodes.new("ShaderNodeTexImage")
                node.image = rough
                node.label = "Roughness"
                node.location = (-460, 0)
                tree.links.new(node.outputs["Color"], bsdf.inputs["Roughness"])

        normal = bpy.data.images.get(ctx.images.get("normal", ""))
        if normal:
            node = tree.nodes.new("ShaderNodeTexImage")
            node.image = normal
            node.label = "Normal"
            node.location = (-720, -300)
            nmap = tree.nodes.new("ShaderNodeNormalMap")
            nmap.location = (-400, -300)
            tree.links.new(node.outputs["Color"], nmap.inputs["Color"])
            tree.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
        return mat

    def _write_provenance(self, ctx, result):
        # ID properties accept only plain str/int/float/list/dict values
        settings_hash = ctx.flags.get("settings_hash") or _settings_hash(
            ctx.settings, ctx.src_names)
        result["fastbake"] = {
            "version": 1,
            "sources": [str(n) for n in ctx.src_names],
            "run": int(ctx.run),
            "quads": int(ctx.flags.get("actual_quads") or ctx.settings["quads"]),
            "tex": int(ctx.settings["tex_size"]),
            "maps": [str(k) for k in sorted(ctx.images)],
            "settings_hash": str(settings_hash),
            "date": datetime.now().isoformat(timespec="seconds"),
        }
        for name in ctx.src_names:
            src = ctx.obj(name)
            if src is None:
                continue
            current = src.get("fastbake_source_of")
            names = [str(v) for v in current] if current else []
            if result.name not in names:
                names.append(result.name)
            src["fastbake_source_of"] = names

    def _stash_sources(self, ctx):
        scene = bpy.context.scene
        # (name, library) key: only a LOCAL collection may be the stash — a
        # library-linked collection with the same name must not be matched
        coll = bpy.data.collections.get((SOURCES_COLLECTION, None))
        if coll is None:
            coll = bpy.data.collections.new(SOURCES_COLLECTION)
        # identity check: a linked collection sharing the name must not make
        # us skip linking the local stash into the scene
        if not any(c is coll for c in scene.collection.children_recursive):
            scene.collection.children.link(coll)

        for name in ctx.src_names:
            src = ctx.obj(name)
            if src is None:
                continue
            if not any(o is src for o in coll.objects):
                coll.objects.link(src)
            for other in list(src.users_collection):
                if other is not coll:
                    other.objects.unlink(src)
        bpy.context.view_layer.update()

        for name in ctx.src_names:
            src = ctx.obj(name)
            if src is None:
                continue
            try:
                src.hide_set(True)
            except RuntimeError:
                src.hide_viewport = True  # not in this view layer
            src.hide_render = True
        ctx.log("moved %d source object(s) to %s"
                % (len(ctx.src_names), SOURCES_COLLECTION))

    def _remove_work_object(self, ctx):
        # retopo keeps the scaffold mesh datablock alive for gate Redo; free
        # it now that the pipeline is committed to this result
        kept = bpy.data.meshes.get(ctx.flags.get("retopo_kept_mesh", ""))
        if kept is not None and kept.users == 0:
            bpy.data.meshes.remove(kept)
        work = ctx.obj(ctx.work_name)
        if work is None:
            return
        mesh = work.data if work.type == "MESH" else None
        bpy.data.objects.remove(work)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        ctx.log("removed scaffold object " + str(ctx.work_name))

    def _select_result(self, result):
        view_layer = bpy.context.view_layer
        for obj in view_layer.objects:
            obj.select_set(False)
        try:
            result.select_set(True)
            view_layer.objects.active = result
        except RuntimeError:
            pass  # result hidden from the view layer — leave selection alone

    def summary(self, ctx):
        return "Material %s assigned to %s; originals stored in %s" % (
            ctx.material_name, ctx.result_name, SOURCES_COLLECTION)


classes = ()
