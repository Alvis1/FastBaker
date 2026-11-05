# Bake stage: normal, color and roughness maps from the source objects onto
# the retopologized result (Cycles selected-to-active).
#
# Roughness is always baked; the pack stage decides whether it ends up in the
# color texture's alpha or stays a separate map. When preflight flagged the
# sources as alpha-driven a fourth, transient "alpha" pass is baked as well
# (plan §2 ruling "the original alpha wins the alpha channel"): the source
# materials are rewired so Principled *Alpha* drives an Emission shader and an
# EMIT bake captures it. The pack stage folds that map into the color image's
# alpha channel and drops the datablock.
#
# Images are created at tex_size * ssaa and downscaled later by the pack stage
# (final names from the start). The result object gets a temporary bake
# material whose image node is both selected and active (5.0 bake-target rule);
# source materials are only touched by the emission rewires (metallic color
# pass, alpha pass) and always restored afterwards.

import time

import bpy
from mathutils.bvhtree import BVHTree

from .. import compat
from .context import Stage, StageError

MAPS = ("normal", "color", "roughness")

BAKE_MAT_NAME = "FB_bake_tmp"
BAKE_NODE_NAME = "FB_bake_target"
TRI_MOD_NAME = "FB_Triangulate"
# The alpha pass never reaches disk (pack folds it into the color alpha and
# removes it), so it is a scratch datablock rather than a named artifact.
ALPHA_IMG_NAME = "FB_alpha_tmp"
# stand-in for sources with no material during the alpha pass (see
# _ensure_opaque_slots): no material == black EMIT == alpha 0
OPAQUE_MAT_NAME = "FB_alpha_opaque"

TRI_NOTICE = ("Result keeps a fixed-diagonal Triangulate modifier so exported "
              "tangents match the bake")

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


def _gpu_compute_available():
    """True when Cycles is actually configured for GPU compute.

    Plan §5 stage 4 asks for "GPU on" for the data passes, but flipping
    cycles.device to 'GPU' without a configured compute device makes Cycles
    fall back to CPU at best and refuse the bake at worst — so we only switch
    when the Cycles preferences report a compute device type AND at least one
    non-CPU device enabled under it.
    """
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return False
    device_type = getattr(prefs, "compute_device_type", "NONE")
    if not device_type or device_type == "NONE":
        return False
    devices = ()
    try:
        devices = prefs.get_devices_for_type(device_type)
    except Exception:
        devices = getattr(prefs, "devices", ())
    return any(getattr(d, "use", False) and getattr(d, "type", "CPU") != "CPU"
               for d in devices)


class BakeStage(Stage):
    stage_id = "bake"
    label = "Baking textures"
    gate = True
    adjust_props = ("margin_px", "ssaa", "normal_y")

    # safe defaults in case cancel()/poll() is reached before start()
    _finished = True
    _cleaned = True
    _cancelled = False
    _rewires = ()
    _rewired = False
    _opaque_slots = ()
    _opaque_mat_name = None

    # -- lifecycle -------------------------------------------------------------

    def start(self, ctx):
        # plan §2 ruling: an alpha-driven source keeps its own alpha in the
        # color texture, so bake it as a fourth, transient map (appended last:
        # pack consumes it right after the three real maps)
        self._maps = list(MAPS)
        if ctx.flags.get("source_uses_alpha"):
            self._maps.append("alpha")
        self._index = 0
        self._finished = False
        self._cleaned = False
        self._map_done_flag = False
        self._map_cancel_flag = False
        self._seen_running = False
        self._map_started_at = 0.0
        self._handlers_installed = False
        self._h_complete = None
        self._h_cancel = None
        self._scene_snapshot = None
        self._rewires = []
        self._rewired = False
        self._used_emission = False
        self._opaque_slots = []
        self._opaque_mat_name = None
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
        # margin dilates in pixels of the SSAA bake image, so scale it by the
        # bake resolution: it then survives the pack stage's downscale as
        # margin_px per 1K of the final texture
        self._margin = int(adv["margin_px"] * self._big / 1024)
        self._normal_g = "NEG_Y" if adv["normal_y"] == "DIRECTX" else "POS_Y"
        # pseudo-progress: rough per-map duration guess, scales with pixel count
        self._est_secs = max(4.0, 8.0 * (self._big / 2048.0) ** 2)

        self._force_visible(sources + [result])
        self._measure_ray_distances(ctx, sources, result)
        self._create_images(ctx)
        self._snapshot_scene(ctx)
        self._configure_scene(ctx)
        self._setup_bake_material(ctx, result)

        # A gate Redo re-enters start() with the modifier still on the result
        # (the success path keeps it, decision 5): reuse it, never stack one.
        mod = (result.modifiers.get(TRI_MOD_NAME)
               or result.modifiers.new(TRI_MOD_NAME, "TRIANGULATE"))
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
            self._cleanup(ctx, success=True)
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
        running = compat.bake_job_running()
        if running:
            self._seen_running = True
        # the job may start suspended (another render-lock job is active), so
        # "not running" only means done once the job was actually seen running
        if self._map_done_flag or (self._seen_running and not running
                                   and elapsed > 1.0):
            key = self._maps[self._index]
            self._post_map(ctx, key)
            ctx.log("baked %s map" % key)
            self._index += 1
            if self._index >= n:
                self._cleanup(ctx, success=True)
                self._finished = True
                return None
            self._launch_map(ctx, self._index)
            return self._index / n
        if not self._seen_running and elapsed > 30.0:
            raise StageError("Bake job failed to start (%s map)"
                             % self._maps[self._index])

        guess = min(0.9, elapsed / self._est_secs)
        return (self._index + guess) / n

    def cancel(self, ctx):
        """Non-blocking (dev/CONTRACTS.md §2 + fix contract decision 3).

        There is no API to abort a running Cycles bake, so this must never
        sleep: the sleep can only run on the main thread, which is the very
        thread that would deliver the job's completion — it just delays the
        cancel. The driver instead keeps polling while
        compat.bake_job_running() is True and calls us once the job drained,
        so no datablock is freed under a live job. _cleanup() itself frees
        nothing (rollback owns that); it only restores scene settings, the
        source materials' rewires and the source visibility.
        """
        self._cancelled = True
        self._remove_handlers()
        if compat.bake_job_running():
            ctx.log("bake: cancelled while a Cycles bake job is still running")
        self._cleanup(ctx)

    def summary(self, ctx):
        s = "Baked %s at %dpx" % (", ".join(self._maps), self._tex)
        if self._ssaa > 1:
            s += " (%dx SSAA)" % self._ssaa
        s += ", ray dist %.4f" % self._ray
        if self._used_emission:
            s += ", color via emission (metallic sources)"
        if "alpha" in self._maps:
            s += ", source alpha baked"
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
        x1.5 -> cage_extrusion. Rays start from the surface inflated outward
        by cage_extrusion and are traced inward, so max_ray_distance must
        cover extrusion + gap: cage + 2x distance (a plain 2x misses)."""
        deps = bpy.context.evaluated_depsgraph_get()
        deps.update()
        trees = []
        for src in sources:
            try:
                # BVHTree.FromObject builds in the object's local space; keep
                # matrix_world so hit points can be measured in world units
                trees.append((BVHTree.FromObject(src, deps),
                              src.matrix_world.inverted(),
                              src.matrix_world.copy()))
            except Exception as exc:
                ctx.log("bake: BVH build failed for %s: %s" % (src.name, exc))
        if not trees:
            raise StageError("Bake: could not build BVH on any source object")

        mw = result.matrix_world
        max_d = 0.0
        for v in result.data.vertices:
            world = mw @ v.co
            best = None
            for tree, inv, src_mw in trees:
                loc, _nrm, _idx, _dist = tree.find_nearest(inv @ world)
                if loc is None:
                    continue
                # find_nearest's dist is local-space; measure the hit in world
                # space (cage_extrusion/max_ray_distance are world units), so
                # parent/inherited scale cannot skew the extrusion
                d = ((src_mw @ loc) - world).length
                if best is None or d < best:
                    best = d
            if best is not None and best > max_d:
                max_d = best
        self._cage = max(max_d * 1.5, 1e-4)
        # documented deviation from plan §5 stage 4 ("x2 = max_ray_distance"):
        # the plan's 2x is measured from the un-inflated surface, so with a
        # 1.5x cage it would leave only 0.5x max_d of reach past the low-poly
        # and miss the far side — cage + 2x max_d is the same guarantee, safe.
        self._ray = max(self._cage + max_d * 2.0, 1e-4)
        ctx.log("bake: surface distance %.5f -> extrusion %.5f, ray %.5f"
                % (max_d, self._cage, self._ray))

    def _create_images(self, ctx):
        for key in self._maps:
            # every delivered artifact name comes from the context (naming
            # templates); the alpha pass is scratch and never saved
            name = ALPHA_IMG_NAME if key == "alpha" else ctx.tex_name(key)
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
        # plan §5 stage 4: data passes run with GPU on — but only when the
        # user actually has a compute device configured (see
        # _gpu_compute_available); cycles.device is in the restore snapshot.
        if _gpu_compute_available():
            try:
                scene.cycles.device = "GPU"
                ctx.log("bake: using GPU compute")
            except (TypeError, AttributeError) as exc:
                ctx.log("bake: could not switch to GPU (%s), staying on CPU" % exc)
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
        elif key == "alpha":
            # source Alpha is rewired into an Emission shader (see _pre_map)
            args.update(type="EMIT")
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
            self._rewire_emission(ctx, "Base Color")
        elif key == "alpha":
            self._rewire_emission(ctx, "Alpha")

    def _post_map(self, ctx, key):
        if self._rewired:
            self._restore_rewires(ctx)

    def _launch_map(self, ctx, index):
        key = self._maps[index]
        # the viewport stays interactive between maps: re-assert the bake
        # selection so a user click cannot redirect the bake (and use_clear)
        # onto their own objects/images
        result = ctx.obj(ctx.result_name)
        if result is None or result.type != "MESH":
            raise StageError("Bake: result mesh is missing")
        sources = [ctx.obj(n) for n in ctx.src_names]
        sources = [o for o in sources if o is not None and o.type == "MESH"]
        if not sources:
            raise StageError("Bake: no source objects left to bake from")
        self._select_for_bake(sources, result)
        self._pre_map(ctx, key)
        self._map_done_flag = False
        self._map_cancel_flag = False
        self._seen_running = False
        try:
            ret = bpy.ops.object.bake("INVOKE_DEFAULT",
                                      **self._bake_args(ctx, key))
        except RuntimeError as exc:
            raise StageError("Bake failed to start (%s): %s" % (key, exc))
        if "CANCELLED" in ret:
            raise StageError("Bake failed to start (%s map)" % key)
        self._map_started_at = time.time()

    # -- emission rewire (metallic color bake / source alpha bake) ---------------

    def _rewire_emission(self, ctx, socket):
        """Wire each source material's `socket` into a temp Emission ->
        Material Output so an EMIT bake captures it.

        socket="Base Color": albedo of metallic materials (they bake black in
        the diffuse pass). socket="Alpha": the source's own alpha, which the
        pack stage folds into the color texture's alpha channel (plan §2
        ruling). Restore info is stored as node names, never references.
        """
        self._rewires = []
        self._rewired = True
        if socket == "Base Color":
            self._used_emission = True
        if socket == "Alpha":
            self._ensure_opaque_slots(ctx)
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
                if getattr(mat, "library", None) is not None:
                    # library-linked materials are read-only: rewiring one
                    # would fail mid-stage (or edit someone else's asset)
                    if socket == "Alpha":
                        # an un-rewired material bakes BLACK under EMIT, i.e.
                        # alpha 0 — the object would silently vanish. Bake the
                        # area opaque through the local stand-in instead.
                        self._swap_linked_slots(ctx, mat)
                        ctx.notice("Material %s is library-linked, so its alpha "
                                   "was baked as opaque — make it local and "
                                   "re-bake for a real cutout" % mat.name)
                    else:
                        ctx.notice("Material %s is library-linked, so its %s "
                                   "could not be baked — make it local and "
                                   "re-bake if that area comes out wrong"
                                   % (mat.name, socket.lower()))
                    continue
                rec = self._rewire_one(ctx, mat, socket)
                if rec:
                    self._rewires.append(rec)

    def _ensure_opaque_slots(self, ctx):
        """Sources with no material bake black under EMIT — i.e. alpha 0, a
        silently vanished object. Give every empty slot (and every mesh with no
        slot at all) a temporary white Emission material for the alpha pass."""
        mat = None
        for name in ctx.src_names:
            obj = ctx.obj(name)
            if obj is None or obj.type != "MESH":
                continue
            empty = [i for i, s in enumerate(obj.material_slots) if s.material is None]
            if not obj.material_slots:
                empty = [None]          # append a slot instead of filling one
            if not empty:
                continue
            if mat is None:
                mat = self._make_opaque_material(ctx)
            for index in empty:
                if index is None:
                    obj.data.materials.append(mat)
                    self._opaque_slots.append(
                        (obj.name, len(obj.data.materials) - 1, True))
                else:
                    obj.material_slots[index].material = mat
                    self._opaque_slots.append((obj.name, index, False))

    def _swap_linked_slots(self, ctx, linked):
        """Point every source slot that uses the read-only `linked` material
        at the opaque stand-in for the alpha pass; _restore_opaque_slots puts
        the linked material back (the linked ID itself is never touched)."""
        mat = (bpy.data.materials.get((self._opaque_mat_name, None))
               if self._opaque_mat_name else None)
        if mat is None:
            mat = self._make_opaque_material(ctx)
        swaps = getattr(self, "_linked_swaps", None)
        if swaps is None:
            swaps = self._linked_swaps = []
        lib = linked.library.filepath
        for name in ctx.src_names:
            obj = ctx.obj(name)
            if obj is None or obj.type != "MESH":
                continue
            for index, slot in enumerate(obj.material_slots):
                if slot.material is linked:
                    swaps.append((obj.name, index, linked.name, lib))
                    slot.material = mat

    def _make_opaque_material(self, ctx):
        mat = bpy.data.materials.new(OPAQUE_MAT_NAME)
        ctx.track_material(mat)
        if mat.node_tree is None:
            mat.use_nodes = True
        tree = mat.node_tree
        out = tree.get_output_node("CYCLES") or tree.nodes.new("ShaderNodeOutputMaterial")
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        tree.links.new(emission.outputs["Emission"], out.inputs["Surface"])
        self._opaque_mat_name = mat.name
        return mat

    def _restore_opaque_slots(self, ctx):
        for name, index, mat_name, lib in reversed(getattr(self, "_linked_swaps", [])):
            obj = bpy.data.objects.get(name)
            mat = bpy.data.materials.get((mat_name, lib))
            if (obj is not None and obj.type == "MESH" and mat is not None
                    and index < len(obj.material_slots)):
                obj.material_slots[index].material = mat
        self._linked_swaps = []
        for name, index, appended in reversed(self._opaque_slots):
            obj = bpy.data.objects.get(name)
            if obj is None or obj.type != "MESH":
                continue
            try:
                if appended:
                    # pop() would leave a permanent empty slot on a mesh that
                    # had none before; clear() restores the original state
                    if len(obj.data.materials) <= 1:
                        obj.data.materials.clear()
                    else:
                        obj.data.materials.pop(index=index)
                elif index < len(obj.material_slots):
                    obj.material_slots[index].material = None
            except Exception as exc:
                ctx.log("bake: could not restore material slot %s[%d]: %s"
                        % (name, index, exc))
        self._opaque_slots = []
        # (name, None): only the LOCAL stand-in we created, never a
        # library-linked datablock that happens to share the name
        mat = (bpy.data.materials.get((self._opaque_mat_name, None))
               if self._opaque_mat_name else None)
        if mat is not None and mat.users == 0:
            bpy.data.materials.remove(mat)
        self._opaque_mat_name = None

    def _rewire_one(self, ctx, mat, socket):
        tree = mat.node_tree
        use_nodes = None
        if tree is None:
            # pre-5.0 node-less material: an EMIT bake of it is black, so give
            # it a temporary node tree and put use_nodes back on restore
            use_nodes = mat.use_nodes
            mat.use_nodes = True
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
        src_in = principled.inputs.get(socket) if principled else None
        if src_in is not None and src_in.is_linked:
            tree.links.new(src_in.links[0].from_socket, emission.inputs["Color"])
        elif socket == "Alpha":
            # constant Alpha — or no Principled at all (a transparent-BSDF mix
            # and the like): bake the value, and default to OPAQUE rather than
            # to a black EMIT that would delete the object from the result
            const = tree.nodes.new("ShaderNodeValue")
            if src_in is not None:
                const.outputs[0].default_value = float(src_in.default_value)
            else:
                const.outputs[0].default_value = float(mat.diffuse_color[3])
                ctx.log("bake: %s has no Principled Alpha input, alpha baked "
                        "as %.3f" % (mat.name, const.outputs[0].default_value))
            tree.links.new(const.outputs[0], emission.inputs["Color"])
            rgb_name = const.name
        else:
            rgb = tree.nodes.new("ShaderNodeRGB")
            if src_in is not None:
                rgb.outputs[0].default_value = tuple(src_in.default_value)
            else:
                rgb.outputs[0].default_value = tuple(mat.diffuse_color)
            tree.links.new(rgb.outputs[0], emission.inputs["Color"])
            rgb_name = rgb.name
        tree.links.new(emission.outputs["Emission"], surface)
        return dict(mat=mat.name, out=out.name, from_node=from_name,
                    from_idx=from_idx, emission=emission.name, rgb=rgb_name,
                    use_nodes=use_nodes)

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
            if rec.get("use_nodes") is not None:
                try:
                    mat.use_nodes = rec["use_nodes"]
                except Exception:
                    pass
        self._rewires = []
        self._rewired = False
        if self._opaque_slots or self._opaque_mat_name:
            self._restore_opaque_slots(ctx)

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

    def _cleanup(self, ctx, success=False):
        """Restore everything except the bake material and images (the pack and
        material stages consume those). Safe to call more than once.

        `success` keeps the Triangulate modifier on the result: plan §5 stage 4
        requires it for export, because MikkTSpace tangents are only consistent
        for the one triangulation the normal map was baked against. On every
        failure path the result object is deleted by the rollback anyway, so
        the modifier is dropped there to keep redo clean.
        """
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
            if mod and success:
                if TRI_NOTICE not in ctx.notices:
                    ctx.notice(TRI_NOTICE)
            elif mod:
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
