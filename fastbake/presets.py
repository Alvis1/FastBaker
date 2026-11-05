# JSON preset system: save / load / export / import.
# Blender's native preset framework stores executable .py files and has no
# import/export, so FastBake keeps plain JSON in the extension user dir
# (plan §8). The schema is the one documented in the plan:
#   {"fastbake_preset": 1, "name": ..., <settings>,
#    "naming": {"template", "material", "style"}, "advanced": {...}}

import copy
import json
import os

import bpy
from bpy.props import StringProperty

from . import compat
from .props import (default_settings_dict, dict_to_settings, naming_block,
                    settings_to_dict)

PRESET_VERSION = 1

# Factory presets are OVERRIDES on top of the property defaults; the files
# written from them always contain every key (see ensure_factory_presets).
FACTORY = {
    "Props 2k": {"quads": 5000, "tex_size": "2048"},
    "Hero 4k": {"quads": 20000, "tex_size": "4096"},
    "Mobile 1k": {"quads": 1500, "tex_size": "1024",
                  "advanced": {"naming_style": "LETTERS"}},
    # plan §6: "a bundled Unreal preset switches to letter suffixes"
    # (Epic mandates the T_/M_ prefixes the default templates already use)
    "Unreal": {"quads": 5000, "tex_size": "2048",
               "advanced": {"naming_style": "LETTERS",
                            "normal_y": "DIRECTX"}},
}


def preset_dir():
    return compat.extension_user_dir("presets")


def sanitize_name(name):
    """Filename-safe preset name. The sanitised form IS the preset name —
    save/load/delete and the list all agree on it."""
    return "".join(c for c in str(name) if c.isalnum() or c in " _-").strip()


def _path(name):
    directory = preset_dir()
    filename = "%s.json" % name
    # a file copied into the presets folder by hand keeps its own spelling:
    # list_presets() shows that stem, so load/delete must find it again
    if filename == os.path.basename(filename) and not os.path.isabs(filename):
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    safe = sanitize_name(name)
    if not safe:
        raise ValueError("A preset name needs at least one letter or number")
    return os.path.join(directory, safe + ".json")


def list_presets():
    try:
        files = sorted(os.listdir(preset_dir()))
    except OSError:
        return []
    return [os.path.splitext(f)[0] for f in files if f.endswith(".json")]


def preset_data(name, settings):
    data = {"fastbake_preset": PRESET_VERSION, "name": name}
    data.update(settings_to_dict(settings))
    data["naming"] = naming_block(data)   # plan §8 schema block
    return data


def write_preset_data(name, data):
    """Store a preset dict verbatim. Returns (stem, overwrote_existing)."""
    path = _path(name)
    stem = os.path.splitext(os.path.basename(path))[0]
    existed = os.path.isfile(path)
    data = dict(data)
    data["fastbake_preset"] = PRESET_VERSION
    data["name"] = stem
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return stem, existed


def save_preset(name, settings):
    """Write a preset. Returns (stem, overwrote_existing).

    `stem` is the name the preset is known by afterwards — sanitised, so the
    caller can report exactly what the user will see in the list.
    """
    path = _path(name)
    stem = os.path.splitext(os.path.basename(path))[0]
    existed = os.path.isfile(path)
    data = preset_data(stem, settings)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return stem, existed


def load_preset(name, settings):
    """Apply a preset file. Returns the list of keys that did not apply."""
    with open(_path(name), "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("fastbake_preset") != PRESET_VERSION:
        raise ValueError("Unsupported preset version")
    return dict_to_settings(data, settings)


def factory_preset_data(name):
    """A COMPLETE preset dict for one factory preset.

    Every setting is written, not just the overrides: dict_to_settings only
    touches keys that are present, so a partial preset would silently leave
    the previous preset's values in place (a Mobile 1k -> Props 2k switch
    keeping letter suffixes, say). FACTORY itself is never mutated — it is
    deep-copied first, or a second call in the same session (add-on
    disable/enable) would regenerate a stripped preset.
    """
    overrides = copy.deepcopy(FACTORY[name])
    try:
        data = default_settings_dict()
    except Exception:
        data = {"advanced": {}}   # RNA not available (unregistered module)
    adv_overrides = overrides.pop("advanced", {}) if isinstance(overrides, dict) else {}
    data.update(overrides)
    data.setdefault("advanced", {}).update(adv_overrides)
    data["naming"] = naming_block(data)
    data["fastbake_preset"] = PRESET_VERSION
    data["name"] = name
    return data


def ensure_factory_presets():
    existing = set(list_presets())
    for name in FACTORY:
        if name in existing:
            continue
        with open(_path(name), "w", encoding="utf-8") as f:
            json.dump(factory_preset_data(name), f, indent=2)


def _report_rejected(operator, rejected):
    if not rejected:
        return
    shown = ", ".join(rejected[:4])
    if len(rejected) > 4:
        shown += ", ... (%d more)" % (len(rejected) - 4)
    operator.report({"WARNING"}, "Preset values not applied: " + shown)


# bpy requires Python to keep a reference to the strings returned by an
# EnumProperty items callback, or Blender may misbehave or even crash.
_enum_item_cache = []


def _preset_items(self, context):
    items = [(n, n, "") for n in list_presets()]
    _enum_item_cache[:] = items or [("NONE", "No presets", "")]
    return _enum_item_cache


class FASTBAKE_OT_preset_save(bpy.types.Operator):
    bl_idname = "fastbake.preset_save"
    bl_label = "Save Preset"
    bl_description = "Save current FastBake settings as a named preset"

    name: StringProperty(name="Name", default="My preset")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        try:
            stem, existed = save_preset(self.name, context.scene.fastbake)
        except ValueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except OSError as exc:
            self.report({"ERROR"}, "Could not save preset: %s" % exc)
            return {"CANCELLED"}
        if existed:
            self.report({"WARNING"}, "Replaced preset '%s'" % stem)
        elif stem != self.name.strip():
            self.report({"INFO"}, "Saved preset as '%s'" % stem)
        else:
            self.report({"INFO"}, "Saved preset '%s'" % stem)
        return {"FINISHED"}


class FASTBAKE_OT_preset_load(bpy.types.Operator):
    bl_idname = "fastbake.preset_load"
    bl_label = "Load Preset"
    bl_description = "Load a FastBake preset"
    bl_property = "preset"

    preset: bpy.props.EnumProperty(items=_preset_items)

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if self.preset == "NONE":
            return {"CANCELLED"}
        try:
            rejected = load_preset(self.preset, context.scene.fastbake)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, "Could not load preset: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Loaded preset '%s'" % self.preset)
        _report_rejected(self, rejected)
        return {"FINISHED"}


class FASTBAKE_OT_preset_delete(bpy.types.Operator):
    bl_idname = "fastbake.preset_delete"
    bl_label = "Delete Preset"
    bl_property = "preset"

    preset: bpy.props.EnumProperty(items=_preset_items)

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        if self.preset == "NONE":
            return {"CANCELLED"}
        try:
            os.remove(_path(self.preset))
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        return {"FINISHED"}


class FASTBAKE_OT_preset_export(bpy.types.Operator):
    bl_idname = "fastbake.preset_export"
    bl_label = "Export Preset"
    bl_description = "Export current settings to a .json file"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        self.filepath = "fastbake_preset.json"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        path = self.filepath
        if not path.lower().endswith(".json"):
            path += ".json"
        data = preset_data(os.path.splitext(os.path.basename(path))[0],
                           context.scene.fastbake)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            self.report({"ERROR"}, "Export failed: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Exported %s" % os.path.basename(path))
        return {"FINISHED"}


class FASTBAKE_OT_preset_import(bpy.types.Operator):
    bl_idname = "fastbake.preset_import"
    bl_label = "Import Preset"
    bl_description = "Import settings from a .json preset file (also saves it to your presets)"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("fastbake_preset") != PRESET_VERSION:
                raise ValueError("Not a FastBake preset")
            rejected = dict_to_settings(data, context.scene.fastbake)
            name = data.get("name") or os.path.splitext(
                os.path.basename(self.filepath))[0]
            # store the FILE, not a re-serialization of the scene: anything
            # this version could not apply stays in the preset instead of
            # being overwritten with the current value
            stem, _existed = write_preset_data(name, data)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, "Import failed: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Imported preset '%s'" % stem)
        _report_rejected(self, rejected)
        return {"FINISHED"}


class FASTBAKE_MT_presets(bpy.types.Menu):
    bl_idname = "FASTBAKE_MT_presets"
    bl_label = "Presets"

    def draw(self, context):
        layout = self.layout
        layout.operator("fastbake.preset_load", icon="FILE_TICK")
        layout.operator("fastbake.preset_save", icon="ADD")
        layout.operator("fastbake.preset_delete", icon="REMOVE")
        layout.separator()
        layout.operator("fastbake.preset_export", icon="EXPORT")
        layout.operator("fastbake.preset_import", icon="IMPORT")


classes = (FASTBAKE_OT_preset_save, FASTBAKE_OT_preset_load,
           FASTBAKE_OT_preset_delete, FASTBAKE_OT_preset_export,
           FASTBAKE_OT_preset_import, FASTBAKE_MT_presets)
