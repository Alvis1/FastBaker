# Lightmap pipeline: bake DIFFUSE DIRECT+INDIRECT light of the ACTIVE object
# into its own UVs (no selected-to-active), denoise via a compositor OIDN
# pass, optional gamma, save as EXR.

import os
import time

import bpy
import numpy as np

from .. import compat, naming
from .context import Stage, StageError
from .state import STATE


class FASTBAKE_OT_lightmap(bpy.types.Operator):
    bl_idname = "fastbake.lightmap"
    bl_label = "Bake Lightmap"
    bl_description = "Bake a denoised lightmap for the active object into its own UVs"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "MESH" and not STATE["running"]

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        adv = context.scene.fastbake.advanced
        col = self.layout.column()
        col.prop(adv, "lm_size")
        col.prop(adv, "lm_samples")
        col.prop(adv, "lm_gamma")
        col.prop(adv, "lm_denoise")

    def execute(self, context):
        from . import driver, preflight
        # build around the ACTIVE object (what poll advertises), not the
        # selection: make_context would substitute a selected object when
        # the active one is unselected
        ctx, err = preflight.make_lightmap_context(context)
        if ctx is None:
            self.report({"ERROR"}, err or "FastBake: cannot start")
            return {"CANCELLED"}
        if not driver.launch(context, driver.build_lightmap_stages(), ctx):
            self.report({"ERROR"}, "FastBake is already running")
            return {"CANCELLED"}
        return {"FINISHED"}


class LightmapStage(Stage):
    stage_id = "lightmap"
    label = "Baking lightmap"
    gate = False
    adjust_props = ("lm_size", "lm_samples", "lm_gamma", "lm_denoise")

    def start(self, ctx):
        self._done = False
        self._baking = False
        self._cleaned = False
        self._t0 = 0.0
        self._est = 10.0
        self._margin = 4
        self._restore = {}            # plain values only (never bpy IDs)
        self._sel_prev = None         # object names
        self._act_prev = None
        self._temp_nodes = []         # (mat name, node name, prev active node name)
        self._forced_use_nodes = []   # mat names whose use_nodes we flipped on (4.x)
        self._temp_mat = None
        self._temp_mat_appended = False
        self._temp_mat_slots = []
        self._temp_camera = None
        self._comp_tree_created = False
        self._comp_nodes = []
        self._comp_iface_created = False
        self._comp_restore_link = None
        self._use_nodes_prev = None
        self._obj_name = ctx.active_name

        adv = ctx.settings["advanced"]
        lm = dict(size=int(adv["lm_size"]), samples=int(adv["lm_samples"]),
                  gamma=float(adv["lm_gamma"]), denoise=bool(adv["lm_denoise"]))
        ctx.flags["lightmap"] = lm

        obj = ctx.obj(self._obj_name)
        if obj is None or obj.type != "MESH":
            raise StageError("Active mesh object is gone")

        try:
            self._prepare(ctx, obj, lm)
            if bpy.app.background:
                self._bake(lm, invoke=False)
                self._finish(ctx)
                self._done = True
            else:
                self._bake(lm, invoke=True)
                self._baking = True
                self._t0 = time.time()
                self._est = max(10.0, lm["samples"] * (lm["size"] / 1024.0) ** 2 * 0.2)
        except Exception:
            self._cleanup(ctx)
            raise

    def poll(self, ctx):
        if self._done:
            return None
        if self._baking:
            elapsed = time.time() - self._t0
            # grace period: the bake job may not be registered on the very
            # first poll after INVOKE_DEFAULT
            if compat.bake_job_running() or elapsed < 0.5:
                return min(0.85, 0.85 * elapsed / self._est)
            self._baking = False
        self._done = True
        self._finish(ctx)  # denoise + gamma + save; seconds-scale, may block
        return None

    def cancel(self, ctx):
        if self._baking:
            # there is no API to abort a running bake job — wait briefly so
            # cleanup does not pull nodes from under Cycles
            t0 = time.time()
            while compat.bake_job_running() and time.time() - t0 < 3.0:
                time.sleep(0.1)
            if compat.bake_job_running():
                ctx.log("lightmap: bake job still running at cancel")
            self._baking = False
        self._cleanup(ctx)

    def summary(self, ctx):
        lm = ctx.flags.get("lightmap", {})
        return "Lightmap %dx%d baked (%d samples%s)" % (
            lm.get("size", 0), lm.get("size", 0), lm.get("samples", 0),
            ", denoised" if lm.get("denoise") else "")

    # -- setup -----------------------------------------------------------------

    def _prepare(self, ctx, obj, lm):
        scene = bpy.context.scene
        r = scene.render
        ims = r.image_settings
        # full snapshot before any mutation so _cleanup is all-or-nothing
        self._restore = {
            "engine": r.engine,
            "samples": scene.cycles.samples,
            "adaptive": scene.cycles.use_adaptive_sampling,
            "res_x": r.resolution_x,
            "res_y": r.resolution_y,
            "pct": r.resolution_percentage,
            "filepath": r.filepath,
            "use_compositing": r.use_compositing,
            "use_sequencer": r.use_sequencer,
            "media_type": getattr(ims, "media_type", None),
            "file_format": ims.file_format,
            "color_mode": ims.color_mode,
            "color_depth": ims.color_depth,
            "camera": scene.camera.name if scene.camera else None,
            "hide_viewport": obj.hide_viewport,
            "hide_eye": obj.hide_get(),
            "hide_render": obj.hide_render,
        }
        self._sel_prev = [o.name for o in bpy.context.selected_objects]
        act = bpy.context.view_layer.objects.active
        self._act_prev = act.name if act else None

        if obj.mode != "OBJECT":
            with compat.temp_override(active_object=obj, object=obj):
                bpy.ops.object.mode_set(mode="OBJECT")

        # only the active object may be selected: multi-object baking would
        # otherwise try to bake every selected object
        obj.hide_viewport = False
        obj.hide_set(False)
        obj.hide_render = False
        for other in list(bpy.context.selected_objects):
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        adv = ctx.settings["advanced"]
        name = naming.tex_name(ctx.base, "lightmap", ctx.run, adv["naming_style"])
        img = bpy.data.images.new(name, lm["size"], lm["size"],
                                  alpha=False, float_buffer=True)
        # new float images default to the scene-linear colorspace — keep it
        ctx.track_image(img)
        ctx.images["lightmap"] = img.name

        self._prepare_materials(ctx, obj, img)

        r.engine = "CYCLES"
        scene.cycles.samples = lm["samples"]
        scene.cycles.use_adaptive_sampling = False
        self._margin = max(1, int(round(int(adv["margin_px"]) * lm["size"] / 1024.0)))

    def _prepare_materials(self, ctx, obj, img):
        # 5.0 rule: the bake writes only to the image node that is BOTH
        # selected and active, and every material slot needs one
        if len(obj.material_slots) == 0 or any(s.material is None for s in obj.material_slots):
            mat = bpy.data.materials.new("FastBake_LM_" + ctx.base)
            if mat.node_tree is None:
                mat.use_nodes = True
            ctx.track_material(mat)
            self._temp_mat = mat.name
            if len(obj.material_slots) == 0:
                obj.data.materials.append(mat)
                self._temp_mat_appended = True
            else:
                for i, slot in enumerate(obj.material_slots):
                    if slot.material is None:
                        slot.material = mat
                        self._temp_mat_slots.append(i)

        seen = set()
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen:
                continue
            seen.add(mat.name)
            if mat.node_tree is None:
                # 4.x legacy material rendering from diffuse_color: remember
                # to flip use_nodes back off in _cleanup
                mat.use_nodes = True
                self._forced_use_nodes.append(mat.name)
            nodes = mat.node_tree.nodes
            prev_active = nodes.active.name if nodes.active else None
            for node in nodes:
                node.select = False
            node = nodes.new("ShaderNodeTexImage")
            node.image = img
            node.select = True
            nodes.active = node
            self._temp_nodes.append((mat.name, node.name, prev_active))

    def _bake(self, lm, invoke):
        kwargs = dict(type="DIFFUSE", pass_filter={"DIRECT", "INDIRECT"},
                      margin=self._margin, margin_type="ADJACENT_FACES",
                      use_selected_to_active=False, use_clear=True)
        if invoke:
            ret = bpy.ops.object.bake("INVOKE_DEFAULT", **kwargs)
            if "RUNNING_MODAL" not in ret and "FINISHED" not in ret:
                raise StageError("Could not start the lightmap bake job")
        else:
            ret = bpy.ops.object.bake(**kwargs)
            if "FINISHED" not in ret:
                raise StageError("Lightmap bake failed")

    # -- post-bake ---------------------------------------------------------------

    def _finish(self, ctx):
        try:
            img = bpy.data.images.get(ctx.images.get("lightmap") or "")
            if img is None:
                raise StageError("Lightmap image disappeared during the bake")
            lm = ctx.flags["lightmap"]
            if lm["denoise"]:
                self._denoise(ctx, img)
            if abs(lm["gamma"] - 1.0) > 1e-3:
                self._apply_gamma(img, lm["gamma"])
            out_path = os.path.join(ctx.output_dir(), img.name + ".exr")
            img.filepath_raw = out_path
            img.file_format = "OPEN_EXR"
            img.save()
            ctx.files["lightmap"] = out_path
            ctx.log("lightmap saved: " + out_path)
        finally:
            self._cleanup(ctx)

    def _denoise(self, ctx, img):
        scene = bpy.context.scene
        size = img.size[0]

        if scene.camera is None:
            cam_data = bpy.data.cameras.new("FastBake_LM_Camera")
            cam = bpy.data.objects.new("FastBake_LM_Camera", cam_data)
            scene.collection.objects.link(cam)
            scene.camera = cam
            ctx.track_object(cam)
            self._temp_camera = cam.name

        # 4.x needs scene.use_nodes flipped on by ensure_compositor_tree;
        # snapshot it first (5.x uses compositing_node_group instead)
        if not hasattr(scene, "compositing_node_group") and hasattr(scene, "use_nodes"):
            self._use_nodes_prev = scene.use_nodes

        tree, created = compat.ensure_compositor_tree(scene)
        self._comp_tree_created = created

        out_node = None
        for node in tree.nodes:
            if node.bl_idname == "CompositorNodeComposite":
                out_node = node
                break
        if out_node is None:
            try:
                out_node = tree.nodes.new("CompositorNodeComposite")
                self._comp_nodes.append(out_node.name)
            except RuntimeError:
                out_node = None
        if out_node is None:
            # 5.1 removed CompositorNodeComposite: the compositor node group
            # outputs through an interface socket + Group Output node
            # (verified on 5.1.2; candidate for a compat seam)
            if not any(getattr(it, "in_out", None) == "OUTPUT"
                       for it in tree.interface.items_tree):
                tree.interface.new_socket("Image", in_out="OUTPUT",
                                          socket_type="NodeSocketColor")
                self._comp_iface_created = True
            for node in tree.nodes:
                if node.bl_idname == "NodeGroupOutput":
                    out_node = node
                    break
            if out_node is None:
                out_node = tree.nodes.new("NodeGroupOutput")
                self._comp_nodes.append(out_node.name)
        out_sock = out_node.inputs.get("Image")
        if out_sock is None and len(out_node.inputs):
            out_sock = out_node.inputs[0]
        if out_sock is None:
            raise StageError("Could not find a compositor output socket for denoising")

        for link in list(tree.links):
            if link.to_node == out_node and link.to_socket == out_sock:
                self._comp_restore_link = (out_node.name, out_sock.name,
                                           link.from_node.name, link.from_socket.name)
                tree.links.remove(link)
                break

        img_node = tree.nodes.new("CompositorNodeImage")
        img_node.image = img
        denoise = tree.nodes.new("CompositorNodeDenoise")
        self._comp_nodes.extend((img_node.name, denoise.name))
        tree.links.new(img_node.outputs["Image"], denoise.inputs["Image"])
        tree.links.new(denoise.outputs["Image"], out_sock)

        r = scene.render
        # the composite result comes entirely from our Image node, so render
        # the scene itself as cheaply as possible
        try:
            r.engine = "BLENDER_WORKBENCH"
        except Exception:
            scene.cycles.samples = 1
        r.resolution_x = size
        r.resolution_y = size
        r.resolution_percentage = 100
        r.use_compositing = True
        r.use_sequencer = False
        tmp_path = os.path.join(ctx.tmpdir, "lightmap_denoise.exr")
        r.filepath = tmp_path
        compat.set_image_format(r.image_settings, "OPEN_EXR",
                                color_depth="32", color_mode="RGBA")
        bpy.ops.render.render(write_still=True)
        if not os.path.exists(tmp_path):
            raise StageError("Denoise render did not produce an output file")

        tmp = bpy.data.images.load(tmp_path)
        try:
            if tuple(tmp.size) != (size, size):
                raise StageError("Denoise render size mismatch (%dx%d)" % tuple(tmp.size))
            buf = np.empty(size * size * 4, np.float32)
            tmp.pixels.foreach_get(buf)
            img.pixels.foreach_set(buf)
            img.update()
        finally:
            bpy.data.images.remove(tmp)
        ctx.log("lightmap denoised via compositor")

    def _apply_gamma(self, img, gamma):
        w, h = img.size
        buf = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(buf)
        inv = 1.0 / gamma
        for offset in (0, 1, 2):  # RGB strides only, alpha untouched
            ch = buf[offset::4]
            np.power(np.clip(ch, 0.0, None), inv, out=ch)
        img.pixels.foreach_set(buf)
        img.update()

    # -- restore -----------------------------------------------------------------

    def _cleanup(self, ctx):
        if self._cleaned:
            return
        self._cleaned = True
        scene = bpy.context.scene
        obj = ctx.obj(self._obj_name)

        # temp image nodes out of the object's materials
        for mat_name, node_name, prev_active in self._temp_nodes:
            try:
                mat = bpy.data.materials.get(mat_name)
                if not mat or not mat.node_tree:
                    continue
                nodes = mat.node_tree.nodes
                node = nodes.get(node_name)
                if node:
                    nodes.remove(node)
                if prev_active:
                    prev = nodes.get(prev_active)
                    if prev:
                        nodes.active = prev
            except Exception as exc:
                ctx.log("lightmap cleanup (nodes): %s" % exc)

        # 4.x legacy materials whose use_nodes we flipped on to insert the
        # bake node: restore so they render from diffuse_color again
        for mat_name in self._forced_use_nodes:
            try:
                mat = bpy.data.materials.get(mat_name)
                if mat:
                    mat.use_nodes = False
            except Exception as exc:
                ctx.log("lightmap cleanup (use_nodes): %s" % exc)

        # temp bake material
        if self._temp_mat:
            try:
                mat = bpy.data.materials.get(self._temp_mat)
                if obj:
                    if self._temp_mat_appended:
                        # the object had zero slots before; materials.pop
                        # leaves a stale empty slot behind (5.1), clear won't
                        obj.data.materials.clear()
                    for i in self._temp_mat_slots:
                        if (i < len(obj.material_slots)
                                and obj.material_slots[i].material
                                and obj.material_slots[i].material.name == self._temp_mat):
                            obj.material_slots[i].material = None
                if mat:
                    bpy.data.materials.remove(mat)
            except Exception as exc:
                ctx.log("lightmap cleanup (material): %s" % exc)

        # compositor
        try:
            tree = compat.get_compositor_tree(scene)
            if tree:
                if self._comp_tree_created:
                    if hasattr(scene, "compositing_node_group"):
                        scene.compositing_node_group = None
                    bpy.data.node_groups.remove(tree)
                else:
                    for name in self._comp_nodes:
                        node = tree.nodes.get(name)
                        if node:
                            tree.nodes.remove(node)
                    if self._comp_iface_created:
                        for item in list(tree.interface.items_tree):
                            if (getattr(item, "name", "") == "Image"
                                    and getattr(item, "in_out", "") == "OUTPUT"):
                                tree.interface.remove(item)
                                break
                    if self._comp_restore_link:
                        to_name, to_sock, from_name, from_sock = self._comp_restore_link
                        to_node = tree.nodes.get(to_name)
                        from_node = tree.nodes.get(from_name)
                        if to_node and from_node:
                            ts = to_node.inputs.get(to_sock)
                            if ts is None and len(to_node.inputs):
                                ts = to_node.inputs[0]
                            fs = from_node.outputs.get(from_sock)
                            if ts and fs:
                                tree.links.new(fs, ts)
        except Exception as exc:
            ctx.log("lightmap cleanup (compositor): %s" % exc)
        if self._use_nodes_prev is not None and hasattr(scene, "use_nodes"):
            try:
                scene.use_nodes = self._use_nodes_prev
            except Exception:
                pass

        # render settings
        rs = self._restore
        if rs:
            try:
                r = scene.render
                r.engine = rs["engine"]
                scene.cycles.samples = rs["samples"]
                scene.cycles.use_adaptive_sampling = rs["adaptive"]
                r.resolution_x = rs["res_x"]
                r.resolution_y = rs["res_y"]
                r.resolution_percentage = rs["pct"]
                r.filepath = rs["filepath"]
                r.use_compositing = rs["use_compositing"]
                r.use_sequencer = rs["use_sequencer"]
                ims = r.image_settings
                if rs["media_type"] is not None and hasattr(ims, "media_type"):
                    ims.media_type = rs["media_type"]  # must precede file_format on 5.x
                ims.file_format = rs["file_format"]
                ims.color_mode = rs["color_mode"]
                ims.color_depth = rs["color_depth"]
            except Exception as exc:
                ctx.log("lightmap cleanup (render settings): %s" % exc)

            try:
                scene.camera = (bpy.data.objects.get(rs["camera"])
                                if rs["camera"] else None)
            except Exception as exc:
                ctx.log("lightmap cleanup (camera): %s" % exc)

            if obj:
                try:
                    obj.hide_viewport = rs["hide_viewport"]
                    obj.hide_set(rs["hide_eye"])
                    obj.hide_render = rs["hide_render"]
                except Exception as exc:
                    ctx.log("lightmap cleanup (visibility): %s" % exc)

        # temp camera
        if self._temp_camera:
            try:
                cam = bpy.data.objects.get(self._temp_camera)
                if cam:
                    data = cam.data
                    bpy.data.objects.remove(cam)
                    if data and data.users == 0:
                        bpy.data.cameras.remove(data)
            except Exception as exc:
                ctx.log("lightmap cleanup (temp camera): %s" % exc)

        # selection
        if self._sel_prev is not None:
            try:
                for other in list(bpy.context.selected_objects):
                    other.select_set(False)
                for name in self._sel_prev:
                    o = bpy.data.objects.get(name)
                    if o:
                        try:
                            o.select_set(True)
                        except RuntimeError:
                            pass
                act = bpy.data.objects.get(self._act_prev) if self._act_prev else None
                if act:
                    bpy.context.view_layer.objects.active = act
            except Exception as exc:
                ctx.log("lightmap cleanup (selection): %s" % exc)


classes = (FASTBAKE_OT_lightmap,)
