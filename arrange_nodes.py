import bpy
import sys
import math
import datetime
from typing import List, Optional


def _get_from_socket_index(node: bpy.types.Node, node_socket: bpy.types.NodeSocket) -> int:
    for i, socket in enumerate(node.outputs):
        if socket == node_socket:
            return i
    assert False


def _get_to_socket_index(node: bpy.types.Node, node_socket: bpy.types.NodeSocket) -> int:
    for i, socket in enumerate(node.inputs):
        if socket == node_socket:
            return i
    assert False


def _get_node_size(node: bpy.types.Node) -> tuple[float, float]:
    """Return node size, including when dimensions are unavailable."""
    epsilon = 1e-05
    width = node.dimensions.x if node.dimensions.x > epsilon else node.width
    height = node.dimensions.y if node.dimensions.y > epsilon else node.height
    return max(width, epsilon), max(height, epsilon)


def _resolve_overlaps(
        node_tree: bpy.types.NodeTree,
        target_nodes: List[bpy.types.Node],
        target_space: float,
        max_num_iters: int,
) -> int:
    """Resolve intersections with exact separation steps until stable."""
    target_set = set(target_nodes)
    nodes = list(node_tree.nodes)
    margin = max(0.0, target_space)
    max_passes = max(1, min(max_num_iters, 2000))

    for pass_index in range(max_passes):
        moved = False
        for index, node_1 in enumerate(nodes):
            width_1, height_1 = _get_node_size(node_1)
            for node_2 in nodes[index + 1:]:
                if node_1 not in target_set and node_2 not in target_set:
                    continue

                width_2, height_2 = _get_node_size(node_2)
                x_1, y_1 = node_1.location
                x_2, y_2 = node_2.location
                overlap_x = min(x_1 + width_1, x_2 + width_2) - max(x_1, x_2) + margin
                overlap_y = min(y_1, y_2) - max(y_1 - height_1, y_2 - height_2) + margin
                if overlap_x <= 0.0 or overlap_y <= 0.0:
                    continue

                center_x_1 = x_1 + width_1 * 0.5
                center_x_2 = x_2 + width_2 * 0.5
                center_y_1 = y_1 - height_1 * 0.5
                center_y_2 = y_2 - height_2 * 0.5
                move_horizontal = overlap_x <= overlap_y
                direction = (1.0 if center_x_1 >= center_x_2 else -1.0) if move_horizontal else (1.0 if center_y_1 >= center_y_2 else -1.0)
                displacement = overlap_x if move_horizontal else overlap_y
                movable_1 = node_1 in target_set
                movable_2 = node_2 in target_set
                if movable_1 and movable_2:
                    displacement_1 = displacement * 0.5
                    displacement_2 = displacement * 0.5
                elif movable_1:
                    displacement_1 = displacement
                    displacement_2 = 0.0
                else:
                    displacement_1 = 0.0
                    displacement_2 = displacement

                if move_horizontal:
                    node_1.location[0] += direction * displacement_1
                    node_2.location[0] -= direction * displacement_2
                else:
                    node_1.location[1] += direction * displacement_1
                    node_2.location[1] -= direction * displacement_2
                moved = True

        if not moved:
            return pass_index

    return max_passes


def _initialize_layout(
        node_tree: bpy.types.NodeTree,
        target_nodes: List[bpy.types.Node],
        target_space: float,
) -> None:
    """Place connected nodes in left-to-right columns before relaxation."""
    target_set = set(target_nodes)
    ranks = {node: 0 for node in target_nodes}
    max_rank = max(1, len(target_nodes))
    for _ in range(max_rank):
        changed = False
        for link in node_tree.links:
            source = link.from_node
            destination = link.to_node
            if source not in target_set or destination not in target_set:
                continue
            candidate = ranks[source] + 1
            if candidate > ranks[destination]:
                ranks[destination] = candidate
                changed = True
        if not changed:
            break

    columns = {}
    for node in target_nodes:
        columns.setdefault(ranks[node], []).append(node)
    column_x = 0.0
    for rank in sorted(columns):
        column = columns[rank]
        column_width = max(_get_node_size(node)[0] for node in column)
        y = 0.0
        for node in sorted(column, key=lambda item: item.name):
            _, height = _get_node_size(node)
            node.location = (column_x, -y)
            y += height + target_space
        column_x += column_width + target_space


def _arrange_nodes_internal_routine(
        node_tree: bpy.types.NodeTree,
        target_nodes: List[bpy.types.Node],
        max_num_iters: int,
        target_space: float,
        fix_horizontal_location: bool,
        fix_vertical_location: bool,
        fix_overlaps: bool,
        verbose: bool,
        is_second_stage: bool,
) -> int:

    epsilon = 1e-05
    target_space = 2.0 * target_space if not is_second_stage else target_space
    k_horizontal_distance = 0.9 if not is_second_stage else 0.5
    k_vertical_distance = 0.5 if not is_second_stage else 0.05

    # Gauss-Seidel-style iterations
    previous_squared_deltas_sum = sys.float_info.max
    iter_count = 0
    for iter_count in range(max_num_iters):
        squared_deltas_sum = 0.0

        if fix_horizontal_location:
            for link in node_tree.links:
                k_horizontal_distance = 0.9 if not is_second_stage else 0.5
                threshold_factor = 2.0

                x_from = link.from_node.location[0]
                x_to = link.to_node.location[0]
                w_from = link.from_node.width
                signed_space = x_to - x_from - w_from
                C = signed_space - target_space
                grad_C_x_from = -1.0
                grad_C_x_to = 1.0

                # Skip if the distance is sufficiently large
                if C >= target_space * threshold_factor:
                    continue

                lagrange = C / (grad_C_x_from * grad_C_x_from + grad_C_x_to * grad_C_x_to)
                delta_x_from = -lagrange * grad_C_x_from
                delta_x_to = -lagrange * grad_C_x_to

                # Update positions only when the node is in the target node list
                if link.from_node in target_nodes:
                    link.from_node.location[0] += k_horizontal_distance * delta_x_from
                    squared_deltas_sum += k_horizontal_distance * k_horizontal_distance * delta_x_from * delta_x_from

                if link.to_node in target_nodes:
                    link.to_node.location[0] += k_horizontal_distance * delta_x_to
                    squared_deltas_sum += k_horizontal_distance * k_horizontal_distance * delta_x_to * delta_x_to

        if fix_vertical_location:
            socket_offset = 20.0

            for link in node_tree.links:
                from_socket_index = _get_from_socket_index(link.from_node, link.from_socket)
                to_socket_index = _get_to_socket_index(link.to_node, link.to_socket)
                y_from = link.from_node.location[1] - socket_offset * from_socket_index
                y_to = link.to_node.location[1] - socket_offset * to_socket_index
                C = y_from - y_to
                grad_C_y_from = 1.0
                grad_C_y_to = -1.0
                lagrange = C / (grad_C_y_from * grad_C_y_from + grad_C_y_to * grad_C_y_to)
                delta_y_from = -lagrange * grad_C_y_from
                delta_y_to = -lagrange * grad_C_y_to

                # Update positions only when the node is in the target node list
                if link.from_node in target_nodes:
                    link.from_node.location[1] += k_vertical_distance * delta_y_from
                    squared_deltas_sum += k_vertical_distance * k_vertical_distance * delta_y_from * delta_y_from

                if link.to_node in target_nodes:
                    link.to_node.location[1] += k_vertical_distance * delta_y_to
                    squared_deltas_sum += k_vertical_distance * k_vertical_distance * delta_y_to * delta_y_to

        if fix_overlaps and is_second_stage:
            k = 0.9
            margin = 0.5 * target_space

            # Examine each node pair once.
            nodes = list(node_tree.nodes)
            for index, node_1 in enumerate(nodes):
                for node_2 in nodes[index + 1:]:
                    if node_1 not in target_nodes and node_2 not in target_nodes:
                        continue

                    x_1 = node_1.location[0]
                    x_2 = node_2.location[0]
                    w_1 = node_1.width
                    w_2 = node_2.width
                    cx_1 = x_1 + 0.5 * w_1
                    cx_2 = x_2 + 0.5 * w_2
                    rx_1 = 0.5 * w_1 + margin
                    rx_2 = 0.5 * w_2 + margin

                    y_1 = node_1.location[1]
                    y_2 = node_2.location[1]
                    _, h_1 = _get_node_size(node_1)
                    _, h_2 = _get_node_size(node_2)
                    cy_1 = y_1 - 0.5 * h_1
                    cy_2 = y_2 - 0.5 * h_2
                    ry_1 = 0.5 * h_1 + margin
                    ry_2 = 0.5 * h_2 + margin

                    C_x = math.fabs(cx_1 - cx_2) - (rx_1 + rx_2)
                    C_y = math.fabs(cy_1 - cy_2) - (ry_1 + ry_2)

                    # If no collision, just skip
                    if C_x >= 0.0 or C_y >= 0.0:
                        continue

                    # Solve collision for the "easier" direction
                    if C_x > C_y:
                        grad_C_x_1 = 1.0 if cx_1 - cx_2 >= 0.0 else -1.0
                        grad_C_x_2 = -1.0 if cx_1 - cx_2 >= 0.0 else 1.0
                        lagrange = C_x / (grad_C_x_1 * grad_C_x_1 + grad_C_x_2 * grad_C_x_2)
                        delta_x_1 = -lagrange * grad_C_x_1
                        delta_x_2 = -lagrange * grad_C_x_2

                        if node_1 in target_nodes:
                            node_1.location[0] += k * delta_x_1
                            squared_deltas_sum += k * k * delta_x_1 * delta_x_1

                        if node_2 in target_nodes:
                            node_2.location[0] += k * delta_x_2
                            squared_deltas_sum += k * k * delta_x_2 * delta_x_2
                    else:
                        grad_C_y_1 = 1.0 if cy_1 - cy_2 >= 0.0 else -1.0
                        grad_C_y_2 = -1.0 if cy_1 - cy_2 >= 0.0 else 1.0
                        lagrange = C_y / (grad_C_y_1 * grad_C_y_1 + grad_C_y_2 * grad_C_y_2)
                        delta_y_1 = -lagrange * grad_C_y_1
                        delta_y_2 = -lagrange * grad_C_y_2

                        if node_1 in target_nodes:
                            node_1.location[1] += k * delta_y_1
                            squared_deltas_sum += k * k * delta_y_1 * delta_y_1

                        if node_2 in target_nodes:
                            node_2.location[1] += k * delta_y_2
                            squared_deltas_sum += k * k * delta_y_2 * delta_y_2

        if verbose:
            print("Iteration #" + str(iter_count) + ": " + str(previous_squared_deltas_sum - squared_deltas_sum))

        # Check the termination conditiion
        if math.fabs(previous_squared_deltas_sum - squared_deltas_sum) < epsilon:
            break

        previous_squared_deltas_sum = squared_deltas_sum

    return iter_count


def arrange_nodes(node_tree: bpy.types.NodeTree,
                  target_nodes: Optional[List[bpy.types.Node]] = None,
                  use_current_layout_as_initial_guess: bool = False,
                  max_num_iters: int = 500,
                  target_space: float = 50.0,
                  fix_horizontal_location: bool = True,
                  fix_vertical_location: bool = True,
                  fix_overlaps: bool = True,
                  verbose: bool = False) -> None:
    """Arrange nodes in the target node tree automatically.

    Parameters
    ----------
    target_nodes : List[bpy.types.Node] or None
        A list of target nodes whose positions will be automatically modified. When this list is not speficied, this function will handle all the nodes in the node tree as targets.
    """

    if target_nodes is None:
        target_nodes = []
        for node in node_tree.nodes:
            target_nodes.append(node)

    if verbose:
        print("-----------------")
        print("Target nodes:")
        for node in target_nodes:
            print("- " + node.name)

    if not use_current_layout_as_initial_guess:
        _initialize_layout(node_tree, target_nodes, target_space)

    time_0 = datetime.datetime.now()

    # First pass
    iter_count_1st = _arrange_nodes_internal_routine(node_tree,
                                                     target_nodes,
                                                     max_num_iters,
                                                     target_space,
                                                     fix_horizontal_location=fix_horizontal_location,
                                                     fix_vertical_location=fix_vertical_location,
                                                     fix_overlaps=fix_overlaps,
                                                     verbose=verbose,
                                                     is_second_stage=False)

    time_1 = datetime.datetime.now()

    # Second pass
    iter_count_2nd = _arrange_nodes_internal_routine(node_tree,
                                                     target_nodes,
                                                     max_num_iters,
                                                     target_space,
                                                     fix_horizontal_location=fix_horizontal_location,
                                                     fix_vertical_location=fix_vertical_location,
                                                     fix_overlaps=fix_overlaps,
                                                     verbose=verbose,
                                                     is_second_stage=True)

    # The iterative force pass can leave a small penetration when one node is
    # fixed or when several overlaps are resolved in the same iteration.
    # Finish with an exact pairwise pass so the returned layout has no overlap.
    overlap_passes = _resolve_overlaps(node_tree, target_nodes, target_space, max_num_iters)

    time_2 = datetime.datetime.now()

    if verbose:
        print("Elapsed time (1st pass): {} (#iters = {})".format(time_1 - time_0, iter_count_1st))
        print("Elapsed time (2nd pass): {} (#iters = {})".format(time_2 - time_1, iter_count_2nd))
        print("Overlap resolution passes: {}".format(overlap_passes))
