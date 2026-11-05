# Pack stage: SSAA downscale, color-alpha channel packing and saving the baked
# textures to the output directory.
#
# Who owns the color texture's alpha channel (plan §2 ruling): the source's own
# alpha if it has one (bake produced a transient "alpha" map — it is folded in
# here and its datablock dropped, so no later stage sees it), else roughness if
# "Roughness to alpha" is on, else Cycles' bake coverage mask.
#
# All pixel work is numpy via Image.pixels.foreach_get/foreach_set (orders of
# magnitude faster than pixels[:]). Synchronous: everything happens in start().

import os

import bpy
import numpy as np

from .. import compat
from .context import Stage, StageError


def _image(name):
    """Local-only image lookup: every name here is one FastBake created, so a
    library-linked datablock sharing the name must never be picked up (it is
    read-only — packing or removing it would fail mid-stage)."""
    return bpy.data.images.get((name, None)) if name else None


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
    # save_render leaves the datablock GENERATED with no filepath, so the
    # pixels would be lost on .blend save/reload — repoint it at the file
    # (16-bit PNG round-trips linearly, verified headless on 5.1).
    colorspace = img.colorspace_settings.name
    img.filepath_raw = path
    img.source = "FILE"
    img.colorspace_settings.name = colorspace  # keep Non-Color after repoint


class PackStage(Stage):
    stage_id = "pack"
    label = "Packing textures"
    gate = False

    def start(self, ctx):
        target = int(ctx.settings["tex_size"])
        out_dir = ctx.output_dir()

        # 1. downscale every SSAA-sized bake to the final resolution
        for key, name in ctx.images.items():
            img = _image(name)
            if img is None:
                raise StageError("Baked %s image '%s' is missing" % (key, name))
            if img.size[0] > target:
                _downscale(img, target)
                ctx.log("downscaled %s to %dx%d" % (img.name, img.size[0], img.size[1]))

        # 2. fill the color texture's alpha channel (plan §2 ruling): the
        #    source's own alpha wins it when there is one, otherwise roughness
        #    may fold in. Without either, the channel keeps Cycles' coverage
        #    mask, which is what an opaque asset wants.
        pack = bool(ctx.settings["roughness_to_alpha"]
                    and not ctx.flags.get("source_uses_alpha"))
        color = _image(ctx.images.get("color", ""))
        rough = _image(ctx.images.get("roughness", ""))
        alpha = _image(ctx.images.get("alpha", ""))
        ctx.flags["roughness_packed"] = False
        ctx.flags["source_alpha_packed"] = False

        if alpha is not None and color is None:
            ctx.log("pack: color image missing, source alpha dropped")
        elif alpha is not None and tuple(alpha.size) != tuple(color.size):
            raise StageError("Alpha map is %dx%d but the color map is %dx%d"
                             % (alpha.size[0], alpha.size[1],
                                color.size[0], color.size[1]))
        elif alpha is not None:
            color = _ensure_alpha(color)
            ctx.images["color"] = color.name
            col = _read_pixels(color)
            alp = _read_pixels(alpha)
            # the alpha pass is a grayscale EMIT bake of the source's Alpha
            # input: its R channel IS the alpha, stored linearly
            np.clip(alp[0::4], 0.0, 1.0, out=col[3::4])
            del alp
            color.pixels.foreach_set(col)
            del col
            color.update()
            color.alpha_mode = "CHANNEL_PACKED"
            ctx.flags["source_alpha_packed"] = True
            ctx.log("packed source alpha into color alpha")
            # close the loop on the preflight notice: the user is told the
            # roughness moved out, so also tell them what took the channel
            ctx.notice("The sources' own alpha is stored in the color "
                       "texture's alpha channel (channel-packed).")
        if alpha is not None:
            # transient map: never saved, never handed to the material stage
            bpy.data.images.remove(alpha)
            del ctx.images["alpha"]

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

        # 3. save: color/roughness 8-bit PNG, normal 16-bit PNG
        for key, name in ctx.images.items():
            img = _image(name)
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
        note = ""
        if ctx.flags.get("roughness_packed"):
            note = ", roughness in color alpha"
        elif ctx.flags.get("source_alpha_packed"):
            note = ", source alpha in color alpha"
        return "Saved %d texture%s to %s%s" % (
            n, "s" if n != 1 else "", ctx.output_dir(), note)


classes = ()
