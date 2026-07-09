# Pipeline context and the Stage protocol.
#
# RULES (see dev/CONTRACTS.md):
# * Datablocks are referenced BY NAME (strings). Never store bpy ID references
#   on the context — undo invalidates them.
# * Synchronous stages do all work in start() and return None from poll()
#   immediately. Asynchronous stages (subprocess, bake jobs) return progress
#   floats from poll() until done.
# * Raise StageError("user readable message") for expected failures.

import os
import tempfile

import bpy


class StageError(Exception):
    """Expected, user-facing pipeline failure."""


class PipelineContext:
    def __init__(self, settings_dict, src_names, active_name):
        self.settings = settings_dict          # plain-dict snapshot (props.settings_to_dict)
        self.src_names = list(src_names)       # original selected mesh objects
        self.active_name = active_name         # active source object
        self.base = ""                         # naming base, set by preflight
        self.run = 0                           # shared run counter, set by preflight
        self.work_name = None                  # merged/duplicated scaffold object
        self.result_name = None                # final low-poly object
        self.images = {}                       # map key -> bpy image name
        self.files = {}                        # map key -> saved absolute filepath
        self.material_name = None
        self.created_objects = []              # names, for rollback (in creation order)
        self.created_images = []
        self.created_materials = []
        self.src_visibility = {}               # name -> (hide_viewport, hide_render)
        self.flags = {}                        # free-form: source_uses_alpha, has_metallic,
                                               # engine_used, lightmap (dict), ...
        self.notices = []                      # user-facing notes for the completion report
        self.report_lines = []                 # log
        self.tmpdir = tempfile.mkdtemp(prefix="fastbake_")

    # -- helpers -------------------------------------------------------------

    def obj(self, name):
        return bpy.data.objects.get(name) if name else None

    def log(self, msg):
        self.report_lines.append(str(msg))
        print("[FastBake]", msg)

    def notice(self, msg):
        self.notices.append(str(msg))
        self.log("NOTICE: " + str(msg))

    def track_object(self, obj):
        if obj.name not in self.created_objects:
            self.created_objects.append(obj.name)

    def track_image(self, img):
        if img.name not in self.created_images:
            self.created_images.append(img.name)

    def track_material(self, mat):
        if mat.name not in self.created_materials:
            self.created_materials.append(mat.name)

    def output_dir(self):
        raw = self.settings["advanced"]["output_dir"] or "//textures/"
        path = bpy.path.abspath(raw)
        if not path or path.startswith("//"):
            # unsaved .blend and relative path: fall back to temp
            path = os.path.join(self.tmpdir, "textures")
        os.makedirs(path, exist_ok=True)
        return path

    def snapshot_counts(self):
        return (len(self.created_objects), len(self.created_images),
                len(self.created_materials))

    def rollback_to(self, counts):
        """Delete datablocks created after `counts` (used by stage redo)."""
        n_obj, n_img, n_mat = counts
        for name in self.created_objects[n_obj:]:
            obj = bpy.data.objects.get(name)
            if obj:
                mesh = obj.data if obj.type == "MESH" else None
                bpy.data.objects.remove(obj)
                if mesh and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
        del self.created_objects[n_obj:]
        for name in self.created_images[n_img:]:
            img = bpy.data.images.get(name)
            if img:
                bpy.data.images.remove(img)
        del self.created_images[n_img:]
        for name in self.created_materials[n_mat:]:
            mat = bpy.data.materials.get(name)
            if mat:
                bpy.data.materials.remove(mat)
        del self.created_materials[n_mat:]


class Stage:
    stage_id = "stage"
    label = "Working"          # big progress label, imperative-ish
    gate = True                # participates in review-each-stage gating
    adjust_props = ()          # advanced prop names shown in the Adjust dialog

    def start(self, ctx):
        raise NotImplementedError

    def poll(self, ctx):
        """Return progress 0..1 while running, None when done."""
        return None

    def cancel(self, ctx):
        pass

    def summary(self, ctx):
        return "%s done" % self.stage_id.capitalize()
