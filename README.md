# nodelayout

![Test](https://github.com/yuki-koyama/nodelayout/workflows/Test/badge.svg)

A Blender add-on for automatic layout of nodes.

This fork improves the layout pass for production node graphs. It places
connected nodes from left to right, separates branches vertically, and runs a
final exact overlap-resolution pass so node bodies do not cover one another.

## Example

Before:
![](./docs/before.png)

After:
![](./docs/after.png)

## Goals

### As an Blender Add-On

This repository can be used as a Blender add-on. It provides `Node Auto Layout` panel UI for the node editor.

![](./docs/panel.png)

#### Installation

1. Download or clone this repository.
2. In Blender, open `Edit > Preferences > Add-ons`.
3. Choose `Install...` and select the repository folder or a ZIP archive.
4. Enable **Node Layout**.

The add-on is available in the `Shader Editor`, `Compositor`, and `Texture
Node Editor`. Open the sidebar with `N`, choose the `Layout` tab, and run
`Arrange all nodes` or `Arrange selected nodes`.

The default layout keeps linked nodes flowing from left to right. Nodes in the
same connection column are spaced vertically, and the final pass enforces the
configured `Target space` between node bounds. Existing positions can still be
used as the initial guess with `Use current as initial`.

### As a Python Library

This repository can be used as a Python library for developing other Blender add-ons and scripts. It provides the following function to perform the auto-layout.
```python
def arrange_nodes(node_tree: bpy.types.NodeTree,
                  target_nodes: Optional[List[bpy.types.Node]] = None,
                  use_current_layout_as_initial_guess: bool = False,
                  max_num_iters: int = 500,
                  target_space: float = 50.0,
                  fix_horizontal_location: bool = True,
                  fix_vertical_location: bool = True,
                  fix_overlaps: bool = True,
                  verbose: bool = False) -> None:
    ...
```

## Blender Versions

The original add-on supports Blender `2.80` or later. This improved fork has
been verified with Blender `5.2.0 LTS`.

## Verification

The included test graph runs in background mode with Blender 5.2 and checks
that the resulting node bounds do not overlap. The layout module can also be
used directly from Python as shown above.

## TODOs

- Improve controllability of the panel UI
- Add support of layout frames
- Create documentation
- Use Jacobi-style constraint projection
- Use numpy and vectorization to improve performance
