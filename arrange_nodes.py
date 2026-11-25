import bpy
import sys
import math
import datetime
from collections import defaultdict, deque
from typing import List, Optional, Tuple


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


def _get_node_size(node: bpy.types.Node) -> Tuple[float, float]:
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
    """Place connected nodes in semantic left-to-right branch lanes."""
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

    # The highest fan-in node is normally the material shader. Its linked
    # input order provides the most useful visual ordering for branches.
    fan_in_nodes = []
    for node in target_nodes:
        linked_inputs = [index for index, socket in enumerate(node.inputs) if socket.is_linked]
        if len(linked_inputs) >= 2:
            fan_in_nodes.append((ranks[node], node, linked_inputs))

    lanes = {}
    if fan_in_nodes:
        highest_rank = max(item[0] for item in fan_in_nodes)
        hubs = [item for item in fan_in_nodes if item[0] == highest_rank]
        for _, hub, linked_inputs in hubs:
            input_lane = {index: lane for lane, index in enumerate(sorted(linked_inputs))}
            incoming_lanes = []
            for link in node_tree.links:
                if link.to_node != hub or link.from_node not in target_set:
                    continue
                socket_index = next(index for index, socket in enumerate(hub.inputs) if socket == link.to_socket)
                lane = float(input_lane[socket_index])
                incoming_lanes.append(lane)
                previous = lanes.get(link.from_node)
                lanes[link.from_node] = lane if previous is None else (previous + lane) * 0.5
            if incoming_lanes:
                lanes[hub] = sum(incoming_lanes) / len(incoming_lanes)

    # Propagate branch lanes upstream and downstream. Averaging is intentional
    # for shared nodes such as a UV Map feeding both color and normal branches.
    for _ in range(max_rank + 1):
        changed = False
        for node in sorted(target_nodes, key=lambda item: ranks[item], reverse=True):
            values = []
            for link in node_tree.links:
                if link.from_node == node and link.to_node in lanes:
                    values.append(lanes[link.to_node])
                elif link.to_node == node and link.from_node in lanes:
                    values.append(lanes[link.from_node])
            if not values:
                continue
            lane = sum(values) / len(values)
            if node not in lanes or abs(lanes[node] - lane) > 1e-05:
                lanes[node] = lane
                changed = True
        if not changed:
            break

    default_lane = sum(lanes.values()) / len(lanes) if lanes else 0.0
    for node in target_nodes:
        lanes.setdefault(node, default_lane)

    columns = {}
    for node in target_nodes:
        columns.setdefault(ranks[node], []).append(node)

    max_height = max(_get_node_size(node)[1] for node in target_nodes)
    lane_spacing = max_height + max(0.0, target_space)
    column_x = 0.0
    for rank in sorted(columns):
        column = columns[rank]
        column_width = max(_get_node_size(node)[0] for node in column)
        for node in sorted(column, key=lambda item: item.name):
            _, height = _get_node_size(node)
            lane_y = -lanes[node] * lane_spacing
            node.location = (column_x, lane_y + height * 0.5)
        column_x += column_width + target_space


def _shader_layout(node_tree, nodes, target_space):
    """Return a layered layout for a flat material graph with one shader hub."""
    if node_tree.bl_idname != 'ShaderNodeTree' or any(
            node.parent or node.hide or node.type in {'FRAME', 'REROUTE'} for node in nodes):
        return None

    outputs = [node for node in nodes if node.type == 'OUTPUT_MATERIAL' and node.is_active_output]
    if len(outputs) != 1:
        return None
    output = outputs[0]
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for link in node_tree.links:
        if link.is_valid and not getattr(link, 'is_muted', False):
            incoming[link.to_node].append(link)
            outgoing[link.from_node].append(link)

    if len(incoming[output]) != 1 or incoming[output][0].to_socket.identifier != 'Surface':
        return None
    hub = incoming[output][0].from_node
    branches = {}
    for link in sorted(incoming[hub], key=lambda link: _get_to_socket_index(hub, link.to_socket)):
        if link.from_node not in branches:
            branches[link.from_node] = len(branches)
    if len(branches) < 2:
        return None

    pending = {node: len(incoming[node]) for node in nodes}
    queue = deque(node for node in nodes if not pending[node])
    rank = dict.fromkeys(nodes, 0)
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for link in outgoing[node]:
            child = link.to_node
            rank[child] = max(rank[child], rank[node] + 1)
            pending[child] -= 1
            if not pending[child]:
                queue.append(child)
    if len(ordered) != len(nodes) or ordered[-2:] != [hub, output]:
        return None

    lanes = {node: {lane} for node, lane in branches.items()}
    for node in reversed(ordered[:-2]):
        lane_set = lanes.setdefault(node, set())
        for link in outgoing[node]:
            lane_set.update(lanes.get(link.to_node, ()))
        if not lane_set:
            return None
    sizes = {node: _get_node_size(node) for node in nodes}
    gap = max(1.0, target_space)
    columns = defaultdict(list)
    rows = defaultdict(list)
    for node in ordered:
        # A shared coordinate source sits left of the branch-local sources.
        column = 2 * rank[node]
        if rank[node] == 0 and len(lanes[node]) > 1:
            column -= 1
        columns[column].append(node)
        if node in lanes and len(lanes[node]) == 1:
            rows[min(lanes[node]), column].append(node)

    x_positions = {}
    x = 0.0
    for column in sorted(columns):
        for node in columns[column]:
            x_positions[node] = x
        x += max(sizes[node][0] for node in columns[column]) + 2.5 * gap

    y_positions = {}
    lane_tops = []
    y = 0.0
    for lane in range(len(branches)):
        lane_tops.append(y)
        lane_bottom = y
        for column in columns:
            next_y = y
            for node in rows.get((lane, column), ()):
                y_positions[node] = next_y
                next_y -= sizes[node][1] + gap
            lane_bottom = min(lane_bottom, next_y)
        y = lane_bottom

    # Fit shared nodes and coordinate sources around their consumers. Move
    # only vertically to clear previously placed nodes in the same column.
    y_positions[hub] = y_positions[output] = lane_tops[len(lane_tops) // 2]
    for column in sorted(columns, reverse=True):
        placed = [node for node in columns[column]
                  if node in y_positions and (incoming[node] or node in branches)]
        for node in columns[column]:
            if node in placed:
                continue
            consumers = list(dict.fromkeys(link.to_node for link in outgoing[node]))
            if len(consumers) == 1 and not incoming[node]:
                consumer = consumers[0]
                y = y_positions[consumer] - max(0.0, sizes[consumer][1] - sizes[node][1])
            else:
                centers = [y_positions[child] - sizes[child][1] / 2 for child in consumers]
                y = (min(centers) + max(centers)) / 2 + sizes[node][1] / 2
            for other in sorted(placed, key=y_positions.get, reverse=True):
                other_top = y_positions[other]
                other_bottom = other_top - sizes[other][1]
                if y > other_bottom - gap and y - sizes[node][1] < other_top + gap:
                    y = other_bottom - gap
            y_positions[node] = y
            placed.append(node)

    x_offset = hub.location.x - x_positions[hub]
    y_offset = hub.location.y - y_positions[hub]
    return {node: (x_positions[node] + x_offset, y_positions[node] + y_offset) for node in nodes}


def _generic_layout(node_tree, nodes, target_space):
    """Arrange any flat node graph from sources on the left to sinks on the right."""
    if any(node.parent for node in nodes):
        return None

    active_links = [
        link for link in node_tree.links
        if link.from_node in nodes and link.to_node in nodes
        and link.is_valid and not getattr(link, 'is_muted', False)
    ]
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for link in active_links:
        incoming[link.to_node].append(link.from_node)
        outgoing[link.from_node].append(link.to_node)

    indegree = {node: len(incoming[node]) for node in nodes}
    rank = dict.fromkeys(nodes, 0)
    queue = deque(sorted((node for node in nodes if indegree[node] == 0), key=lambda node: node.name))
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for child in sorted(outgoing[node], key=lambda item: item.name):
            rank[child] = max(rank[child], rank[node] + 1)
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    # A cycle has no meaningful source rank. Keep it in a stable fallback
    # column while still laying out the acyclic portion of the graph.
    remaining = [node for node in nodes if node not in ordered]
    if remaining:
        cycle_rank = max((rank[node] for node in ordered), default=0) + 1
        for node in sorted(remaining, key=lambda item: item.name):
            rank[node] = cycle_rank
            ordered.append(node)

    sizes = {node: _get_node_size(node) for node in nodes}
    gap = max(1.0, target_space)
    columns = defaultdict(list)
    for node in ordered:
        columns[rank[node]].append(node)

    # A barycenter sort keeps each column close to the vertical order of its
    # connected predecessors, which makes the resulting links easier to read.
    y_positions = {}
    for column in sorted(columns):
        column_nodes = columns[column]
        if column == 0:
            column_nodes.sort(key=lambda node: node.name)
        else:
            column_nodes.sort(key=lambda node: (
                sum(y_positions[source] - sizes[source][1] * 0.5 for source in incoming[node])
                / len(incoming[node]) if incoming[node] else float('inf'),
                node.name,
            ))
        y = 0.0
        for node in column_nodes:
            y_positions[node] = y
            y -= sizes[node][1] + gap

    x_positions = {}
    x = 0.0
    for column in sorted(columns):
        for node in columns[column]:
            x_positions[node] = x
        x += max(sizes[node][0] for node in columns[column]) + 2.5 * gap

    min_x = min(x_positions.values())
    max_x = max(x_positions[node] + sizes[node][0] for node in nodes)
    min_y = min(y_positions[node] - sizes[node][1] for node in nodes)
    max_y = max(y_positions.values())
    old_min_x = min(node.location.x for node in nodes)
    old_max_x = max(node.location.x + sizes[node][0] for node in nodes)
    old_min_y = min(node.location.y - sizes[node][1] for node in nodes)
    old_max_y = max(node.location.y for node in nodes)
    offset_x = (old_min_x + old_max_x - min_x - max_x) * 0.5
    offset_y = (old_min_y + old_max_y - min_y - max_y) * 0.5
    return {
        node: (x_positions[node] + offset_x, y_positions[node] + offset_y)
        for node in nodes
    }


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

    nodes = list(node_tree.nodes)
    target_nodes = nodes if target_nodes is None else list(target_nodes)
    if not target_nodes:
        return

    if (not use_current_layout_as_initial_guess and max_num_iters > 0
            and fix_horizontal_location and fix_vertical_location and fix_overlaps
            and set(target_nodes) == set(nodes)):
        positions = _shader_layout(node_tree, nodes, target_space)
        if positions is None:
            positions = _generic_layout(node_tree, nodes, target_space)
        if positions is not None:
            for node, location in positions.items():
                node.location = location
            return

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
    overlap_passes = 0
    if fix_overlaps:
        overlap_passes = _resolve_overlaps(node_tree, target_nodes, target_space, max_num_iters)

    time_2 = datetime.datetime.now()

    if verbose:
        print("Elapsed time (1st pass): {} (#iters = {})".format(time_1 - time_0, iter_count_1st))
        print("Elapsed time (2nd pass): {} (#iters = {})".format(time_2 - time_1, iter_count_2nd))
        print("Overlap resolution passes: {}".format(overlap_passes))
