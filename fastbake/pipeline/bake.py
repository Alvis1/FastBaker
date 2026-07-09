# Bake stage: normal, color and roughness maps from the source objects onto
# the retopologized result (Cycles selected-to-active).
#
# Roughness is always baked; the pack stage decides whether it ends up in the
# color texture's alpha or stays a separate map. Images are created at
# tex_size * ssaa and downscaled later by the pack stage (final names from the
# start). The result object gets a temporary bake material whose image node is
# both selected and active (5.0 bake-target rule); source materials are only
# touched for the metallic emission-rewire during the color bake.

import time

import bpy
from mathutils.bvhtree import BVHTree

from .. import compat, naming
from .context import Stage, StageError

MAPS = ("normal", "color", "roughness")

BAKE_MAT_NAME = "FB_bake_tmp"
BAKE_NODE_NAME = "FB_bake_target"
TRI_MOD_NAME = "FB_Triangulate"

# scene / cycles / bake settings we touch and must restore
_SCENE_KEYS = ("render.engine", "cycles.samples", "cycles.use_adaptive_sampling",
               "cycles.device")
_BAKE_KEYS = ("use_selected_to_active", "use_cage", "use_clear", "target",
              "cage_extrusion", "max_ray_distance", "margin", "margin_type")


def _rget(obj, dotted):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _rset(obj, dotted, value):
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class BakeStage(Stage):
    stage_id = "bake"
    label = "Baking textures"
    gate = True
    adjust_props = ("margin_px", "ssaa", "normal_y")

    # -- lifecycle -------------------------------------------------------------

    def start(self, ctx):
        self._maps = list(MAPS)
        self._index = 0
        self._finished = False
        self._cleaned = False
        self._map_done_flag = False
        self._map_cancel_flag = False
        self._map_started_at = 0.0
        self._handlers_installed = False
        self._h_complete = None
        self._h_cancel = None
        self._scene_snapshot = None
        self._rewires = []
        self._rewired = False
        self._used_emission = False
        self._vis_saved = {}
        self._bake_mat_name = None
        self._cage = 0.0
        self._ray = 0.0
        try:
            self._start_inner(ctx)
        except Exception:
            self._cleanup(ctx)
            raise

    def _start_inner(self, ctx):
        result = ctx.obj(ctx.result_name)
        if result is None or result.type != "MESH":
            raise StageError("Bake: result mesh is missing")
        sources = [ctx.obj(n) for n in ctx.src_names]
        sources = [o for o in sources if o is not None and o.type == "MESH"]
        if not sources:
            raise StageError("Bake: no source objects left to bake from")
        if not result.data.uv_layers:
            raise StageError("Bake: result mesh has no UV map")

        settings = ctx.settings
        adv = settings["advanced"]
        tex = int(settings["tex_size"])
        ssaa = int(adv["ssaa"])
        self._big = tex * ssaa
        self._tex = tex
        self._ssaa = ssaa
        self._margin = int(adv["margin_px"] * tex / 1024)
        self._normal_g = "NEG_Y" if adv["normal_y"] == "DIRECTX" else "POS_Y"
        # pseudo-progress: rough per-map duration guess, scales with pixel count
        self._est_secs = max(4.0, 8.0 * (self._big / 2048.0) ** 2)

        self._force_visible(sources + [result])
        self._measure_ray_distances(ctx, sources, result)
        self._create_images(ctx)
        self._snapshot_scene(ctx)
        self._configure_scene(ctx)
        self._setup_bake_material(ctx, result)

        mod = result.modifiers.new(TRI_MOD_NAME, "TRIANGULATE")
        mod.quad_method = "FIXED"

        self._select_for_bake(sources, result)

        if bpy.app.background:
            # blocking sequential bakes
            for i, key in enumerate(self._maps):
                self._pre_map(ctx, key)
                try:
                    ret = bpy.ops.object.bake(**self._bake_args(ctx, key))
                except RuntimeError as exc:
                    raise StageError("Bake failed (%s): %s" % (key, exc))
                finally:
                    self._post_map(ctx, key)
                if "FINISHED" not in ret:
                    raise StageError("Bake failed (%s map)" % key)
                ctx.log("baked %s map" % key)
            self._cleanup(ctx)
            self._finished = True
            return

        self._install_handlers()
        self._launch_map(ctx, 0)

    def poll(self, ctx):
        if self._finished:
            return None
        try:
            return self._poll_inner(ctx)
        except Exception:
            self._cleanup(ctx)
            raise

    def _poll_inner(self, ctx):
        if self._map_cancel_flag:
            raise StageError("Bake job was cancelled")

        n = len(self._maps)
        elapsed = time.time() - self._map_started_at
        if self._map_done_flag or (not compat.bake_job_running() and elapsed > 1.0):
            key = self._maps[self._index]
            self._post_map(ctx, key)
            ctx.log("baked %s map" % key)
            self._index += 1
            if self._index >= n:
                self._cleanup(ctx)
                self._finished = True
                return None
            self._launch_map(ctx, self._index)
            return self._index / n

        guess = min(0.9, elapsed / self._est_secs)
        return (self._index + guess) / n

    def cancel(self, ctx):
        # no API to abort a running bake job; restore what we can
        self._cleanup(ctx)

    def summary(self, ctx):
        s = "Baked %s at %dpx" % (", ".join(self._maps), self._tex)
        if self._ssaa > 1:
            s += " (%dx SSAA)" % self._ssaa
        s += ", ray dist %.4f" % self._ray
        if self._used_emission:
            s += ", color via emission (metallic sources)"
        return s

    # -- setup helpers ---------------------------------------------------------

    def _force_visible(self, objects):
        for obj in objects:
            try:
                hidden = obj.hide_get()
            except RuntimeError:
                hidden = False
            self._vis_saved[obj.name] = (obj.hide_viewport, obj.hide_render, hidden)
            obj.hide_viewport = False
            obj.hide_render = False
            try:
                obj.hide_set(False)
            except RuntimeError:
                pass

    def _measure_ray_distances(self, ctx, sources, result):
        """Max distance from result vertices to the nearest source surface;
        x1.5 -> cage_extrusion, x2 -> max_ray_distance."""
        deps = bpy.context.evaluated_depsgraph_get()
        deps.update()
        trees = []
        for src in sources:
            try:
                # BVHTree.FromObject builds in the object's local space
                trees.append((BVHTree.FromObject(src, deps),
                              src.matrix_world.inverted()))
            except Exception as exc:
                ctx.log("bake: BVH build failed for %s: %s" % (src.name, exc))
        if not trees:
            raise StageError("Bake: could not build BVH on any source object")

        mw = result.matrix_world
        max_d = 0.0
        for v in result.data.vertices:
            world = mw @ v.co
            best = None
            for tree, inv in trees:
                loc, _nrm, _idx, dist = tree.find_nearest(inv @ world)
                if loc is not None and (best is None or dist < best):
                    best = dist
            if best is not None and best > max_d:
                max_d = best
        self._cage = max(max_d * 1.5, 1e-4)
        self._ray = max(max_d * 2.0, 1e-4)
        ctx.log("bake: surface distance %.5f -> extrusion %.5f, ray %.5f"
                % (max_d, self._cage, self._ray))

    def _create_images(self, ctx):
        style = ctx.settings["advanced"]["naming_style"]
        for key in self._maps:
            name = naming.tex_name(ctx.base, key, ctx.run, style)
            img = bpy.data.images.new(name, width=self._big, height=self._big,
                                      alpha=True,
                                      float_buffer=(key == "normal"))
            img.colorspace_settings.name = "sRGB" if key == "color" else "Non-Color"
            ctx.track_image(img)
            ctx.images[key] = img.name

    def _snapshot_scene(self, ctx):
        scene = bpy.context.scene
        snap = {}
        for key in _SCENE_KEYS:
            try:
                snap[key] = _rget(scene, key)
            except AttributeError:
                pass
        for key in _BAKE_KEYS:
            snap["render.bake." + key] = getattr(scene.render.bake, key)
        self._scene_snapshot = snap

    def _configure_scene(self, ctx):
        scene = bpy.context.scene
        try:
            scene.render.engine = "CYCLES"
        except TypeError:
            raise StageError("Cycles render engine is not available")
        scene.cycles.samples = 2
        scene.cycles.use_adaptive_sampling = False
        sb = scene.render.bake
        sb.use_selected_to_active = True
        sb.use_cage = False
        sb.use_clear = True
        sb.target = "IMAGE_TEXTURES"
        sb.cage_extrusion = self._cage
        sb.max_ray_distance = self._ray
        sb.margin = self._margin
        sb.margin_type = "ADJACENT_FACES"

    def _setup_bake_material(self, ctx, result):
        mat = bpy.data.materials.new(BAKE_MAT_NAME)
        ctx.track_material(mat)
        if mat.node_tree is None:
            mat.use_nodes = True  # pre-5.0; 5.x materials always have a tree
        nodes = mat.node_tree.nodes
        for n in nodes:
            n.select = False
        node = nodes.new("ShaderNodeTexImage")
        node.name = BAKE_NODE_NAME
        # 5.0+ bakes only into image nodes that are BOTH selected and active
        node.select = True
        nodes.active = node
        result.data.materials.clear()
        result.data.materials.append(mat)
        self._bake_mat_name = mat.name

    def _select_for_bake(self, sources, result):
        view_layer = bpy.context.view_layer
        view_layer.objects.active = result
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass
        for obj in view_layer.objects:
            obj.select_set(False)
        for obj in sources:
            obj.select_set(True)
        result.select_set(True)
        view_layer.objects.active = result

    # -- per-map ----------------------------------------------------------------

    def _bake_args(self, ctx, key):
        args = dict(use_selected_to_active=True, use_cage=False, use_clear=True,
                    target="IMAGE_TEXTURES", cage_extrusion=self._cage,
                    max_ray_distance=self._ray, margin=self._margin,
                    margin_type="ADJACENT_FACES")
        if key == "normal":
            args.update(type="NORMAL", normal_space="TANGENT", normal_r="POS_X",
                        normal_g=self._normal_g, normal_b="POS_Z")
        elif key == "color":
            if ctx.flags.get("has_metallic"):
                args.update(type="EMIT")
            else:
                args.update(type="DIFFUSE", pass_filter={"COLOR"})
        else:
            args.update(type="ROUGHNESS")
        return args

    def _pre_map(self, ctx, key):
        mat = bpy.data.materials.get(self._bake_mat_name)
        node = mat.node_tree.nodes.get(BAKE_NODE_NAME) if mat else None
        img = bpy.data.images.get(ctx.images.get(key, ""))
        if node is None or img is None:
            raise StageError("Bake: target image node is missing")
        node.image = img
        if key == "color" and ctx.flags.get("has_metallic"):
            self._rewire_emission(ctx)

    def _post_map(self, ctx, key):
        if key == "color" and self._rewired:
            self._restore_rewires(ctx)

    def _launch_map(self, ctx, index):
        key = self._maps[index]
        self._pre_map(ctx, key)
        self._map_done_flag = False
        self._map_cancel_flag = False
        try:
            ret = bpy.ops.object.bake("INVOKE_DEFAULT",
                                      **self._bake_args(ctx, key))
        except RuntimeError as exc:
            raise StageError("Bake failed to start (%s): %s" % (key, exc))
        if "CANCELLED" in ret:
            raise StageError("Bake failed to start (%s map)" % key)
        self._map_started_at = time.time()

    # -- emission rewire (metallic color bake) -----------------------------------

    def _rewire_emission(self, ctx):
        """Wire each source material's Base Color into a temp Emission ->
        Material Output so an EMIT bake captures albedo of metals too.
        Restore info is stored as node names, never references."""
        self._rewires = []
        self._rewired = True
        self._used_emission = True
        seen = set()
        for name in ctx.src_names:
            obj = ctx.obj(name)
            if obj is None:
                continue
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None or mat.name in seen:
                    continue
                seen.add(mat.name)
                rec = self._rewire_one(ctx, mat)
                if rec:
                    self._rewires.append(rec)

    def _rewire_one(self, ctx, mat):
        tree = mat.node_tree
        if tree is None:
            ctx.log("bake: material %s has no node tree, skipped rewire" % mat.name)
            return None
        out = tree.get_output_node("CYCLES")
        if out is None:
            ctx.log("bake: material %s has no output node, skipped rewire" % mat.name)
            return None
        surface = out.inputs.get("Surface")
        if surface is None:
            return None

        from_name = None
        from_idx = 0
        from_node = None
        if surface.is_linked:
            link = surface.links[0]
            from_node = link.from_node
            from_name = from_node.name
            for i, sock in enumerate(from_node.outputs):
                if sock.identifier == link.from_socket.identifier:
                    from_idx = i
                    break

        principled = None
        if from_node is not None and from_node.bl_idname == "ShaderNodeBsdfPrincipled":
            principled = from_node
        else:
            for n in tree.nodes:
                if n.bl_idname == "ShaderNodeBsdfPrincipled":
                    principled = n
                    break

        emission = tree.nodes.new("ShaderNodeEmission")
        rgb_name = None
        color_in = principled.inputs.get("Base Color") if principled else None
        if color_in is not None and color_in.is_linked:
            tree.links.new(color_in.links[0].from_socket, emission.inputs["Color"])
        else:
            rgb = tree.nodes.new("ShaderNodeRGB")
            if color_in is not None:
                rgb.outputs[0].default_value = tuple(color_in.default_value)
            else:
                rgb.outputs[0].default_value = tuple(mat.diffuse_color)
            tree.links.new(rgb.outputs[0], emission.inputs["Color"])
            rgb_name = rgb.name
        tree.links.new(emission.outputs["Emission"], surface)
        return dict(mat=mat.name, out=out.name, from_node=from_name,
                    from_idx=from_idx, emission=emission.name, rgb=rgb_name)

    def _restore_rewires(self, ctx):
        for rec in self._rewires:
            mat = bpy.data.materials.get(rec["mat"])
            if mat is None or mat.node_tree is None:
                continue
            tree = mat.node_tree
            for node_key in ("emission", "rgb"):
                node = tree.nodes.get(rec[node_key]) if rec[node_key] else None
                if node:
                    tree.nodes.remove(node)
            if rec["from_node"]:
                out = tree.nodes.get(rec["out"])
                src = tree.nodes.get(rec["from_node"])
                if out and src and rec["from_idx"] < len(src.outputs):
                    tree.links.new(src.outputs[rec["from_idx"]],
                                   out.inputs["Surface"])
        self._rewires = []
        self._rewired = False

    # -- handlers -----------------------------------------------------------------

    def _install_handlers(self):
        stage = self

        def on_complete(*_args):
            stage._map_done_flag = True

        def on_cancel(*_args):
            stage._map_cancel_flag = True

        self._h_complete = on_complete
        self._h_cancel = on_cancel
        _pre, complete, cancel = compat.bake_handlers()
        complete.append(on_complete)
        cancel.append(on_cancel)
        self._handlers_installed = True

    def _remove_handlers(self):
        if not self._handlers_installed:
            return
        _pre, complete, cancel = compat.bake_handlers()
        for lst, fn in ((complete, self._h_complete), (cancel, self._h_cancel)):
            try:
                lst.remove(fn)
            except ValueError:
                pass
        self._handlers_installed = False

    # -- teardown -------------------------------------------------------------------

    def _cleanup(self, ctx):
        """Restore everything except the bake material and images (the pack and
        material stages consume those). Safe to call more than once."""
        if self._cleaned:
            return
        self._cleaned = True
        self._remove_handlers()
        try:
            self._restore_rewires(ctx)
        except Exception as exc:
            ctx.log("bake cleanup: rewire restore failed: %s" % exc)

        result = ctx.obj(ctx.result_name)
        if result is not None:
            mod = result.modifiers.get(TRI_MOD_NAME)
            if mod:
                result.modifiers.remove(mod)

        for name, (hv, hr, hidden) in self._vis_saved.items():
            obj = bpy.data.objects.get(name)
            if obj is None:
                continue
            obj.hide_viewport = hv
            obj.hide_render = hr
            try:
                obj.hide_set(hidden)
            except RuntimeError:
                pass
        self._vis_saved = {}

        if self._scene_snapshot:
            scene = bpy.context.scene
            # restore the engine last so cycles/bake props exist while writing
            keys = sorted(self._scene_snapshot, key=lambda k: k == "render.engine")
            for key in keys:
                try:
                    _rset(scene, key, self._scene_snapshot[key])
                except Exception as exc:
                    ctx.log("bake cleanup: could not restore %s: %s" % (key, exc))
            self._scene_snapshot = None


classes = ()
