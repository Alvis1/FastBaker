# Single seam for every Blender 4.x / 5.x API difference FastBake touches.
# Everything version-dependent goes through here — never branch on
# bpy.app.version anywhere else.

import bpy

VERSION = bpy.app.version
IS_50 = VERSION >= (5, 0, 0)
HAS_MANIFOLD_BOOLEAN = VERSION >= (4, 5, 0)


# --- compositor -------------------------------------------------------------
# Blender 5.0 removed scene.node_tree -> scene.compositing_node_group.

def ensure_compositor_tree(scene):
    """Return (tree, created: bool) for the scene compositor."""
    if IS_50:
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new("FastBake Compositing", "CompositorNodeTree")
            scene.compositing_node_group = tree
            return tree, True
        return tree, False
    scene.use_nodes = True
    return scene.node_tree, False


def get_compositor_tree(scene):
    return scene.compositing_node_group if IS_50 else scene.node_tree


def new_gamma_node(tree):
    # 5.0 replaced CompositorNodeGamma with the shader node equivalent.
    try:
        return tree.nodes.new("CompositorNodeGamma")
    except RuntimeError:
        return tree.nodes.new("ShaderNodeGamma")


# --- image saving -----------------------------------------------------------
# 5.0: ImageFormatSettings.media_type must be set before file_format.

def set_image_format(image_settings, file_format, color_depth=None, color_mode=None):
    if IS_50 and hasattr(image_settings, "media_type"):
        image_settings.media_type = "IMAGE"
    image_settings.file_format = file_format
    if color_depth is not None:
        image_settings.color_depth = color_depth
    if color_mode is not None:
        image_settings.color_mode = color_mode


# --- booleans ---------------------------------------------------------------

def boolean_solver(prefer_manifold):
    """Solver enum value for the Boolean modifier."""
    if prefer_manifold and HAS_MANIFOLD_BOOLEAN:
        return "MANIFOLD"
    return "EXACT"


# --- bake job handlers ------------------------------------------------------
# NOTE: the correct handler name is object_bake_cancel (the 3.3 release notes
# misspell it as object_bake_canceled).

def bake_handlers():
    h = bpy.app.handlers
    return h.object_bake_pre, h.object_bake_complete, h.object_bake_cancel


def bake_job_running():
    try:
        return bpy.app.is_job_running("OBJECT_BAKE")
    except Exception:
        return False


# --- misc -------------------------------------------------------------------

def temp_override(**kwargs):
    return bpy.context.temp_override(**kwargs)


def extension_user_dir(subpath=""):
    """Writable per-extension dir; falls back to user config when running as
    a plain module (dev / headless tests)."""
    try:
        return bpy.utils.extension_path_user(__package__, path=subpath, create=True)
    except Exception:
        return bpy.utils.user_resource("CONFIG", path="fastbake/" + subpath, create=True)
