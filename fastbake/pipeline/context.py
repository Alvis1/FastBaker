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
import shutil
import tempfile
import time

import bpy

from .. import compat, naming


class StageError(Exception):
    """Expected, user-facing pipeline failure."""


# Temp/cache lives exclusively under the extension user dir (plan §9 review
# compliance) — never in the OS temp location.
_TMP_PREFIX = "run_"
_TMP_MAX_AGE = 24 * 3600


def tmp_root():
    return compat.extension_user_dir("tmp")


def purge_stale_tmpdirs(max_age=_TMP_MAX_AGE):
    """Remove leftovers of runs that died before teardown (crash, kill).

    Best effort: a missing or unwritable tmp root is not an error.
    """
    try:
        root = tmp_root()
        entries = os.listdir(root)
    except Exception:
        return
    now = time.time()
    for entry in entries:
        if not entry.startswith(_TMP_PREFIX):
            continue
        path = os.path.join(root, entry)
        try:
            if os.path.isdir(path) and now - os.path.getmtime(path) > max_age:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


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
        self.undo_hooks = []                   # callables run (in reverse) by driver rollback
                                               # to restore state rollback_to can't (e.g. the
                                               # transforms applied to source originals)
        # Created on first use only: the BAKE button builds throw-away
        # contexts (re-bake confirm dialog) that must not litter the disk.
        self._tmpdir = None

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

    # -- temp files ------------------------------------------------------------

    @property
    def tmpdir(self):
        """Per-run scratch dir, created on first access (see plan §9)."""
        if self._tmpdir is None:
            try:
                self._tmpdir = tempfile.mkdtemp(prefix=_TMP_PREFIX, dir=tmp_root())
            except Exception:
                # unwritable extension dir: a run is still better than a crash
                self._tmpdir = tempfile.mkdtemp(prefix="fastbake_")
        return self._tmpdir

    def tmpdir_if_created(self):
        """The temp dir path, or None when this context never needed one."""
        return self._tmpdir

    # -- output ---------------------------------------------------------------

    def output_path(self):
        """Absolute output directory, or None when it cannot be resolved.

        None means an unsaved .blend with a //relative path — the UI asks for
        a folder (plan §5 stage 0). There is deliberately NO temp-dir
        fallback: textures the user cannot find are a silent failure.
        """
        raw = self.settings.get("advanced", {}).get("output_dir") or "//textures/"
        if raw.startswith("//") and not bpy.data.filepath:
            return None
        path = bpy.path.abspath(raw)
        if not path or not os.path.isabs(path):
            return None
        return os.path.normpath(path)

    def output_dir(self, create=True):
        path = self.output_path()
        if path is None:
            raise StageError(
                "This .blend has never been saved, so //textures/ has no "
                "location. Save the file or set an output folder in "
                "Advanced ▸ Output.")
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    # -- naming ----------------------------------------------------------------
    # Single place every stage asks for artifact names, so template/style
    # settings are honoured uniformly (see naming.py).

    def tex_name(self, key):
        adv = self.settings.get("advanced", {})
        return naming.tex_name(self.base, key, self.run,
                               adv.get("naming_style", "VERBOSE"),
                               adv.get("naming_template"))

    def mat_name(self):
        adv = self.settings.get("advanced", {})
        return naming.mat_name(self.base, self.run,
                               adv.get("material_template"))

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
