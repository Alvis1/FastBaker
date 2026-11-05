# Single seam for every Blender 4.x / 5.x API difference FastBake touches.
# Everything version-dependent goes through here — never branch on
# bpy.app.version anywhere else.

import contextlib

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


def compositor_use_nodes_snapshot(scene):
    """Value to restore `scene.use_nodes` with after ensure_compositor_tree,
    or None on versions that have no such toggle (5.x drives the compositor
    through scene.compositing_node_group instead)."""
    if hasattr(scene, "compositing_node_group") or not hasattr(scene, "use_nodes"):
        return None
    return scene.use_nodes


def restore_compositor_use_nodes(scene, value):
    if value is None:
        return
    try:
        scene.use_nodes = value
    except Exception:
        pass


def discard_compositor_tree(scene, tree):
    """Detach and delete a tree that ensure_compositor_tree created (5.x only:
    on 4.x the tree is embedded in the scene and never "created")."""
    if hasattr(scene, "compositing_node_group"):
        scene.compositing_node_group = None
    if bpy.data.node_groups.get(tree.name) == tree:
        bpy.data.node_groups.remove(tree)


COMPOSITOR_OUTPUT_SOCKET = "Image"


def ensure_compositor_output(tree):
    """Return (out_node, created_node_names, iface_created) for the node the
    composite result must be linked into.

    4.x/5.0 composite through a CompositorNodeComposite node. 5.1 removed that
    node type: a compositing node group outputs through an OUTPUT interface
    socket plus a Group Output node (verified on 5.1.2 and 5.2.1).
    """
    created = []
    for node in tree.nodes:
        if node.bl_idname == "CompositorNodeComposite":
            return node, created, False
    try:
        node = tree.nodes.new("CompositorNodeComposite")
    except RuntimeError:
        pass
    else:
        created.append(node.name)
        return node, created, False

    iface_created = False
    if not any(getattr(it, "in_out", None) == "OUTPUT"
               for it in tree.interface.items_tree):
        tree.interface.new_socket(COMPOSITOR_OUTPUT_SOCKET, in_out="OUTPUT",
                                  socket_type="NodeSocketColor")
        iface_created = True
    out_node = None
    for node in tree.nodes:
        if node.bl_idname == "NodeGroupOutput":
            out_node = node
            break
    if out_node is None:
        out_node = tree.nodes.new("NodeGroupOutput")
        created.append(out_node.name)
    return out_node, created, iface_created


def remove_compositor_output_socket(tree):
    """Undo the OUTPUT interface socket ensure_compositor_output added."""
    for item in list(tree.interface.items_tree):
        if (getattr(item, "name", "") == COMPOSITOR_OUTPUT_SOCKET
                and getattr(item, "in_out", "") == "OUTPUT"):
            tree.interface.remove(item)
            return


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


def image_format_snapshot(image_settings):
    """Plain-value snapshot of the scene output format (media_type only exists
    on 5.x); restore it with restore_image_format."""
    return {"media_type": getattr(image_settings, "media_type", None),
            "file_format": image_settings.file_format,
            "color_mode": image_settings.color_mode,
            "color_depth": image_settings.color_depth}


def restore_image_format(image_settings, snap):
    if not snap:
        return
    # media_type must be written before file_format on 5.x
    if snap.get("media_type") is not None and hasattr(image_settings, "media_type"):
        image_settings.media_type = snap["media_type"]
    image_settings.file_format = snap["file_format"]
    image_settings.color_mode = snap["color_mode"]
    image_settings.color_depth = snap["color_depth"]


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


def cycles_available():
    """True when the Cycles engine is registered. Cycles ships as an add-on
    and can be disabled, and its engine id is not in the static RNA enum, so
    look for the registered RenderEngine subclass."""
    try:
        for cls in bpy.types.RenderEngine.__subclasses__():
            if getattr(cls, "bl_idname", "") == "CYCLES":
                return True
    except Exception:
        pass
    # fall back to the scene property the Cycles add-on registers
    return hasattr(bpy.types.Scene, "cycles")


def bake_job_running():
    try:
        return bpy.app.is_job_running("OBJECT_BAKE")
    except Exception:
        return False


# --- UV selection -----------------------------------------------------------

@contextlib.contextmanager
def uv_select_sync(tool_settings):
    """Force UV sync selection + FACE mesh select mode for the block, then
    restore both.

    5.x dropped the unsynced UV selection mode (BMLoopUV.select is gone), so
    UV operators report through the mesh selection; on 4.x sync selection is
    optional and has to be switched on explicitly. FACE select mode is
    required for exact per-face counts: in VERT mode the selection flush
    promotes every face whose verts are selected, inflating the counts.
    """
    old_sync = getattr(tool_settings, "use_uv_select_sync", None)
    try:
        old_mode = tuple(tool_settings.mesh_select_mode)
    except Exception:
        old_mode = None
    try:
        try:
            if old_sync is not None:
                tool_settings.use_uv_select_sync = True
            tool_settings.mesh_select_mode = (False, False, True)
        except Exception:
            pass  # read-only in some versions/contexts: proceed as-is
        yield
    finally:
        try:
            if old_sync is not None:
                tool_settings.use_uv_select_sync = old_sync
            if old_mode is not None:
                tool_settings.mesh_select_mode = old_mode
        except Exception:
            pass


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


# --- path properties --------------------------------------------------------------
# 5.x warns on every '//'-relative assignment to a DIR_PATH/FILE_PATH property
# unless it declares PATH_SUPPORTS_BLEND_RELATIVE; 4.5 has no such option item.

def path_prop_options():
    """Options set for StringProperties that hold blend-relative paths."""
    try:
        bpy.props.StringProperty(subtype="DIR_PATH",
                                 options={"PATH_SUPPORTS_BLEND_RELATIVE"})
        return {"PATH_SUPPORTS_BLEND_RELATIVE"}
    except (TypeError, ValueError):
        return set()
