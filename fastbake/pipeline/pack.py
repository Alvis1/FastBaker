# Pack stage: SSAA downscale, roughness -> color-alpha channel packing, and
# saving the baked textures to the output directory.
#
# All pixel work is numpy via Image.pixels.foreach_get/foreach_set (orders of
# magnitude faster than pixels[:]). Synchronous: everything happens in start().

import os

import bpy
import numpy as np

from .. import compat
from .context import Stage, StageError


def _read_pixels(img):
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    return buf


def _downscale(img, target):
    """Box-mean downscale of a square SSAA image to target x target."""
    w, h = img.size
    if w <= target:
        return
    factor = w // target
    if factor < 2 or w != target * factor or h % factor:
        # unexpected size — let Blender's own filter handle it
        img.scale(target, target)
        img.update()
        return
    buf = _read_pixels(img)
    # flat -> (rows, factor, cols, factor, rgba); mean over the 2x2 (or NxN) blocks
    small = buf.reshape(h // factor, factor, w // factor, factor, 4).mean(axis=(1, 3))
    del buf
    img.scale(w // factor, h // factor)
    img.pixels.foreach_set(small.reshape(-1))
    del small
    img.update()


def _ensure_alpha(img):
    """Return an RGBA version of img under the same name.

    Images created with alpha=False (depth 24/96) silently premultiply and drop
    the alpha channel on save — rebuild as RGBA before packing into alpha.
    """
    if img.depth in (32, 128):
        return img
    name = img.name
    w, h = img.size
    buf = _read_pixels(img)
    new = bpy.data.images.new(name + ".rgba", w, h, alpha=True,
                              float_buffer=img.is_float)
    # colorspace must be set BEFORE pixels: changing it reloads the buffer
    new.colorspace_settings.name = img.colorspace_settings.name
    new.pixels.foreach_set(buf)
    del buf
    new.update()
    bpy.data.images.remove(img)
    new.name = name  # old name freed by the remove above; tracking still matches
    return new


def _save_byte_png(img, path):
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()


def _save_16bit_png(img, path):
    """16-bit RGB PNG via scene render settings + save_render.

    Verified headless on 5.1: Non-Color float images round-trip linearly (no
    view transform applied) and the file is 16-bit quantized.
    """
    scene = bpy.context.scene
    settings = scene.render.image_settings
    snap = {"file_format": settings.file_format,
            "color_depth": settings.color_depth,
            "color_mode": settings.color_mode}
    if hasattr(settings, "media_type"):
        snap["media_type"] = settings.media_type
    try:
        compat.set_image_format(settings, "PNG", color_depth="16", color_mode="RGB")
        img.save_render(path, scene=scene)
    finally:
        if "media_type" in snap:
            settings.media_type = snap["media_type"]
        settings.file_format = snap["file_format"]
        try:
            settings.color_depth = snap["color_depth"]
            settings.color_mode = snap["color_mode"]
        except TypeError:
            pass  # restored format may not accept the stored depth/mode combo


class PackStage(Stage):
    stage_id = "pack"
    label = "Packing textures"
    gate = False

    def start(self, ctx):
        target = int(ctx.settings["tex_size"])
        out_dir = ctx.output_dir()

        # 1. downscale every SSAA-sized bake to the final resolution
        for key, name in ctx.images.items():
            img = bpy.data.images.get(name)
            if img is None:
                raise StageError("Baked %s image '%s' is missing" % (key, name))
            if img.size[0] > target:
                _downscale(img, target)
                ctx.log("downscaled %s to %dx%d" % (img.name, img.size[0], img.size[1]))

        # 2. pack roughness into the color texture's alpha channel
        pack = bool(ctx.settings["roughness_to_alpha"]
                    and not ctx.flags.get("source_uses_alpha"))
        color = bpy.data.images.get(ctx.images.get("color", ""))
        rough = bpy.data.images.get(ctx.images.get("roughness", ""))
        if pack and color and rough:
            color = _ensure_alpha(color)
            ctx.images["color"] = color.name
            col = _read_pixels(color)
            rgh = _read_pixels(rough)
            col[3::4] = rgh[0::4]  # grayscale roughness R -> alpha, stored linearly
            del rgh
            color.pixels.foreach_set(col)
            del col
            color.update()
            color.alpha_mode = "CHANNEL_PACKED"
            bpy.data.images.remove(rough)
            del ctx.images["roughness"]
            ctx.flags["roughness_packed"] = True
            ctx.log("packed roughness into color alpha")
        else:
            ctx.flags["roughness_packed"] = False

        # 3. save: color/roughness 8-bit PNG, normal 16-bit PNG
        for key, name in ctx.images.items():
            img = bpy.data.images.get(name)
            if img is None:
                raise StageError("Baked %s image '%s' is missing" % (key, name))
            path = os.path.join(out_dir, img.name + ".png")
            try:
                if key == "normal":
                    _save_16bit_png(img, path)
                else:
                    _save_byte_png(img, path)
            except (RuntimeError, OSError) as exc:
                raise StageError("Could not save %s: %s" % (path, exc))
            ctx.files[key] = path
            ctx.log("saved " + path)

    def summary(self, ctx):
        n = len(ctx.files)
        note = ", roughness in color alpha" if ctx.flags.get("roughness_packed") else ""
        return "Saved %d texture%s to %s%s" % (
            n, "s" if n != 1 else "", ctx.output_dir(), note)


classes = ()
