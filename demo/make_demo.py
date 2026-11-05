# Build demo/FastBake_demo.blend — the demo asset for plan §11 M7
# ("factory presets, demo assets, listing") and the scene M5's "denoised
# lightmap visually clean at 64 samples on demo scene" is judged on.
#
# The scene is a sculpt-like procedural rock: a dense noise-displaced
# icosphere with a flattened base, two materials (bare stone + moss on the
# upward faces) and two smaller chunks wedged into it, so a FastBake run
# exercises multi-object merge, multi-material baking and a real high-poly
# -> low-poly reduction. Nothing is hand-modelled: re-running this script
# reproduces the file byte-for-byte modulo Blender's own versioning.
#
# Run:
#   blender --background --factory-startup --python demo/make_demo.py
#   blender --background --factory-startup --python demo/make_demo.py -- \
#       --out /tmp/other.blend --subdivisions 4
#
# The file is saved with the three source objects selected and the rock
# active, so opening it and pressing BAKE in the FastBake panel just works.

import argparse
import math
import os
import sys

import bpy
from mathutils import Vector, noise

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "FastBake_demo.blend")


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images,
                  bpy.data.cameras, bpy.data.lights):
        for item in list(block):
            if item.users == 0:
                block.remove(item)


def make_material(name, color, roughness, bump=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    if bump:
        # A procedural noise bump: gives the normal bake something real to
        # capture beyond the silhouette.
        tex = mat.node_tree.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = 18.0
        tex.inputs["Detail"].default_value = 6.0
        bump_node = mat.node_tree.nodes.new("ShaderNodeBump")
        bump_node.inputs["Strength"].default_value = bump
        mat.node_tree.links.new(tex.outputs["Fac"], bump_node.inputs["Height"])
        mat.node_tree.links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def sculpt(obj, amp, freq, offset, flatten=None):
    """Multi-octave Perlin displacement along the normals (a sculpt, in 8
    lines). `flatten` clamps the underside so the rock sits on the ground."""
    off = Vector(offset)
    for v in obj.data.vertices:
        n = v.normal if v.normal.length > 1e-6 else Vector((0.0, 0.0, 1.0))
        p = v.co * freq + off
        d = (noise.noise(p) * amp
             + noise.noise(p * 2.1) * amp * 0.45
             + noise.noise(p * 4.7) * amp * 0.2)
        v.co = v.co + n * d
        if flatten is not None and v.co.z < flatten:
            v.co.z = flatten - (flatten - v.co.z) * 0.15
    obj.data.update()
    for poly in obj.data.polygons:
        poly.use_smooth = True


def add_rock(name, subdivisions, radius, location, amp, freq, seed_offset,
             flatten=None):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions,
                                          radius=radius, location=location)
    obj = bpy.context.active_object
    obj.name = name
    sculpt(obj, amp * radius, freq / radius, seed_offset, flatten)
    return obj


def assign_moss(obj, moss_index, threshold=0.45):
    """Second material slot on the faces that point upward — the mossy top."""
    for poly in obj.data.polygons:
        if poly.normal.z > threshold:
            poly.material_index = moss_index


def build(subdivisions):
    clear_scene()
    stone = make_material("DemoStone", (0.28, 0.26, 0.24, 1.0), 0.82, bump=0.35)
    moss = make_material("DemoMoss", (0.16, 0.30, 0.11, 1.0), 0.95, bump=0.6)

    rock = add_rock("DemoRock", subdivisions, 1.0, (0.0, 0.0, 0.0),
                    amp=0.26, freq=1.7, seed_offset=(3.7, 1.1, 8.3),
                    flatten=-0.55)
    rock.data.materials.append(stone)
    rock.data.materials.append(moss)
    assign_moss(rock, 1)

    chunk_a = add_rock("DemoChunkA", max(2, subdivisions - 1), 0.42,
                       (0.78, 0.25, 0.18), amp=0.3, freq=2.4,
                       seed_offset=(0.4, 6.2, 2.9))
    chunk_a.rotation_euler = (0.6, 0.2, 1.1)
    chunk_a.data.materials.append(stone)

    chunk_b = add_rock("DemoChunkB", max(2, subdivisions - 2), 0.3,
                       (-0.55, -0.42, 0.55), amp=0.34, freq=3.1,
                       seed_offset=(9.1, 0.8, 4.4))
    chunk_b.rotation_euler = (1.2, 0.0, 0.4)
    chunk_b.data.materials.append(stone)
    chunk_b.data.materials.append(moss)
    assign_moss(chunk_b, 1, threshold=0.2)

    # A camera and a sun: the lightmap stage wants a camera, and the scene
    # should render on sight for screenshots/listing images.
    bpy.ops.object.camera_add(location=(3.6, -3.6, 2.4),
                              rotation=(math.radians(66), 0.0, math.radians(45)))
    bpy.context.scene.camera = bpy.context.active_object
    bpy.ops.object.light_add(type="SUN", location=(2.5, -1.5, 4.0),
                             rotation=(math.radians(35), 0.0, math.radians(30)))
    bpy.context.active_object.data.energy = 3.0

    # Ready to bake: sources selected, the rock active.
    bpy.ops.object.select_all(action="DESELECT")
    for obj in (rock, chunk_a, chunk_b):
        obj.select_set(True)
    bpy.context.view_layer.objects.active = rock
    return rock, chunk_a, chunk_b


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="make_demo")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--subdivisions", type=int, default=6,
                        help="icosphere subdivisions for the main rock "
                             "(6 = 20480 tris, the intended demo density; "
                             "drop to 5 for a lighter file)")
    args = parser.parse_args(argv)

    objs = build(args.subdivisions)
    tris = sum(len(o.data.polygons) for o in objs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.out),
                                compress=True)
    print("[make_demo] wrote %s (%d faces across %d objects)"
          % (args.out, tris, len(objs)))


if __name__ == "__main__":
    main()
