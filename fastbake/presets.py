# JSON preset system: save / load / export / import.
# Blender's native preset framework stores executable .py files and has no
# import/export, so FastBake keeps plain JSON in the extension user dir.

import json
import os

import bpy
from bpy.props import StringProperty

from . import compat
from .props import dict_to_settings, settings_to_dict

PRESET_VERSION = 1

FACTORY = {
    "Props 2k": {"quads": 5000, "tex_size": "2048"},
    "Hero 4k": {"quads": 20000, "tex_size": "4096"},
    "Mobile 1k": {"quads": 1500, "tex_size": "1024",
                  "advanced": {"naming_style": "LETTERS"}},
}


def preset_dir():
    return compat.extension_user_dir("presets")


def _path(name):
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip()
    return os.path.join(preset_dir(), safe + ".json")


def list_presets():
    try:
        files = sorted(os.listdir(preset_dir()))
    except OSError:
        return []
    return [os.path.splitext(f)[0] for f in files if f.endswith(".json")]


def save_preset(name, settings):
    data = {"fastbake_preset": PRESET_VERSION, "name": name}
    data.update(settings_to_dict(settings))
    with open(_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_preset(name, settings):
    with open(_path(name), "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("fastbake_preset") != PRESET_VERSION:
        raise ValueError("Unsupported preset version")
    dict_to_settings(data, settings)


def ensure_factory_presets():
    existing = set(list_presets())
    for name, overrides in FACTORY.items():
        if name in existing:
            continue
        data = {"fastbake_preset": PRESET_VERSION, "name": name,
                "quads": 5000, "tex_size": "2048",
                "roughness_to_alpha": True, "review_stages": False,
                "advanced": {}}
        adv = overrides.pop("advanced", {}) if isinstance(overrides, dict) else {}
        data.update(overrides)
        data["advanced"].update(adv)
        with open(_path(name), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


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
        if not self.name.strip():
            self.report({"ERROR"}, "Preset needs a name")
            return {"CANCELLED"}
        save_preset(self.name, context.scene.fastbake)
        self.report({"INFO"}, "Saved preset '%s'" % self.name)
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
            load_preset(self.preset, context.scene.fastbake)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, "Could not load preset: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Loaded preset '%s'" % self.preset)
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
        except OSError as exc:
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
        data = {"fastbake_preset": PRESET_VERSION, "name": "exported"}
        data.update(settings_to_dict(context.scene.fastbake))
        path = self.filepath
        if not path.lower().endswith(".json"):
            path += ".json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
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
            dict_to_settings(data, context.scene.fastbake)
            name = data.get("name") or os.path.splitext(os.path.basename(self.filepath))[0]
            save_preset(name, context.scene.fastbake)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.report({"ERROR"}, "Import failed: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Imported preset '%s'" % name)
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
