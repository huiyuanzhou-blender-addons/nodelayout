# blender test-nodes.blend --background --python test.py

import bpy
import sys
import os
from itertools import combinations

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
import nodelayout

material = bpy.data.materials["Material"]
nodelayout.arrange_nodes(material.node_tree, verbose=True)


def test_material_layout():
    material = bpy.data.materials.new("Layout regression")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputMaterial")
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    uv = tree.nodes.new("ShaderNodeUVMap")
    local_uv = tree.nodes.new("ShaderNodeUVMap")
    base, packed, normal = (tree.nodes.new("ShaderNodeTexImage") for _ in range(3))
    split = tree.nodes.new("ShaderNodeSeparateColor")
    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    for texture in (base, normal):
        tree.links.new(uv.outputs["UV"], texture.inputs["Vector"])
    tree.links.new(local_uv.outputs["UV"], packed.inputs["Vector"])
    tree.links.new(packed.outputs["Color"], split.inputs["Color"])
    tree.links.new(normal.outputs["Color"], normal_map.inputs["Color"])
    tree.links.new(base.outputs["Color"], shader.inputs["Base Color"])
    tree.links.new(split.outputs[0], shader.inputs["Metallic"])
    tree.links.new(split.outputs[1], shader.inputs["Roughness"])
    tree.links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])
    tree.links.new(shader.outputs[0], output.inputs["Surface"])

    gap = 50.0
    nodelayout.arrange_nodes(tree, target_space=gap)
    assert base.location.x == packed.location.x == normal.location.x
    assert base.location.y > packed.location.y > normal.location.y
    assert split.location.x == normal_map.location.x
    assert split.location.y == packed.location.y
    assert normal_map.location.y == normal.location.y
    assert (uv.location.x < local_uv.location.x < base.location.x
            < split.location.x < shader.location.x < output.location.x)
    for first, second in combinations(tree.nodes, 2):
        ax, ay = first.location
        bx, by = second.location
        assert (ax + first.width + gap <= bx or bx + second.width + gap <= ax
                or ay - first.height - gap >= by or by - second.height - gap >= ay)
    positions = {node.name: tuple(node.location) for node in tree.nodes}
    nodelayout.arrange_nodes(tree, target_space=gap)
    assert positions == {node.name: tuple(node.location) for node in tree.nodes}

    for link in list(tree.links):
        if link.from_node in (uv, local_uv):
            tree.links.remove(link)
    tree.nodes.remove(uv)
    tree.nodes.remove(local_uv)
    nodelayout.arrange_nodes(tree, target_space=gap)
    assert base.location.x == packed.location.x == normal.location.x
    assert base.location.y > packed.location.y > normal.location.y


def test_generic_layout():
    tree = bpy.data.node_groups.new("Generic layout regression", "ShaderNodeTree")
    source = tree.nodes.new("ShaderNodeTexNoise")
    ramp = tree.nodes.new("ShaderNodeValToRGB")
    math_node = tree.nodes.new("ShaderNodeMath")
    vector = tree.nodes.new("ShaderNodeVectorMath")
    loose = tree.nodes.new("ShaderNodeValue")
    tree.links.new(source.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], math_node.inputs[0])
    tree.links.new(source.outputs["Fac"], vector.inputs[0])

    nodelayout.arrange_nodes(tree, target_space=40.0)
    assert source.location.x < ramp.location.x < math_node.location.x
    assert source.location.x < vector.location.x
    for first, second in combinations(tree.nodes, 2):
        ax, ay = first.location
        bx, by = second.location
        assert (ax + first.width + 40.0 <= bx or bx + second.width + 40.0 <= ax
                or ay - first.height - 40.0 >= by or by - second.height - 40.0 >= ay)
    positions = {node.name: tuple(node.location) for node in tree.nodes}
    nodelayout.arrange_nodes(tree, target_space=40.0)
    assert positions == {node.name: tuple(node.location) for node in tree.nodes}


if bpy.app.version >= (5, 2, 0):
    test_material_layout()
    test_generic_layout()

import pathlib

pathlib.Path("success").touch()
