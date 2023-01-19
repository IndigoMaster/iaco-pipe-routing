"""
Grid class definition.
"""

import logging
import math
from copy import deepcopy
from typing import List

import numpy as np

from pipe_router.bounding_box import BoundingBox
from pipe_router.graph import UndirectedGraph
from pipe_router.pipe_route import PipeRoute
from pipe_router.point3 import Point3

NEIGHBOR_DIRECTIONS = [
    Point3(-1, 0, 0),
    Point3(1, 0, 0),
    Point3(0, -1, 0),
    Point3(0, 1, 0),
    Point3(0, 0, -1),
    Point3(0, 0, 1)
]


class Grid:
    """
    Represents a 3D grid with categorized nodes and various node operations.
    """

    class NodeTypes:
        """
        Enum class; defines supported node types
        """
        INVALID = 0
        SUPERIOR = 1
        GENERAL = 2

        @staticmethod
        def node_eval(node_type: int) -> float:
            """
            Gets the evaluation value of the given node type.
            Refer to Equation 1 in paper.

            :param node_type: node type (enum value)
            :return: evaluation value
            """
            if node_type == Grid.NodeTypes.GENERAL:
                return 0.5
            elif node_type == Grid.NodeTypes.INVALID:
                return 1
            elif node_type == Grid.NodeTypes.SUPERIOR:
                return 0
            else:
                raise ValueError(f'Unknown node type: {node_type}')

    def __init__(self, true_3d_size: Point3, unit_grid_size: float):
        """
        Builds a Grid object.

        :param true_3d_size: true size of enclosing volume as a point with X, Y, Z sizes
        :param unit_grid_size: size of unit grid (cell size); assumed cubical
        """
        self.true_3d_size = true_3d_size
        self.unit_grid_size = unit_grid_size
        self.node_count_x = math.floor(self.true_3d_size.x / unit_grid_size) + 1
        self.node_count_y = math.floor(self.true_3d_size.y / unit_grid_size) + 1
        self.node_count_z = math.floor(self.true_3d_size.z / unit_grid_size) + 1
        self.total_node_count = self.node_count_x * self.node_count_y * self.node_count_z
        self.occupied_spaces: List[BoundingBox] = []
        self.free_spaces: List[BoundingBox] = []
        self.volume_true: BoundingBox = BoundingBox(Point3.zero(), true_3d_size)
        self.volume_grid: BoundingBox = self.resize_box_to_grid(self.volume_true)
        self.graph: UndirectedGraph = UndirectedGraph()

    def add_occupied_box(self, bb: BoundingBox) -> None:
        """
        Adds an occupied space into the grid. The bounding box is enlarged to fit the grid.
        The original bounding box is not modified.

        :param bb: occupied space bounding box
        """
        bb_ = self.resize_box_to_grid(bb)
        self.occupied_spaces.append(bb_)

    def add_free_space_box(self, bb: BoundingBox):
        """
        Adds a free space into the grid. The bounding box is enlarged to fit the grid.
        The original bounding box is not modified.

        :param bb: free space bounding box
        """
        bb_ = self.resize_box_to_grid(bb)
        self.free_spaces.append(bb_)

    def resize_box_to_grid(self, bb: BoundingBox) -> BoundingBox:
        """
        Constructs a bounding box that resized to exactly fit the grid
        (i.e. discrete coordinates). The original bounding box is not modified.

        :param bb: bounding box to resize
        :return: resized bounding box
        """
        # lower point is rounded down
        grid_pt1 = self.xyz_to_grid(bb.p1, rounding='floor')
        # upper point is rounded up
        grid_pt2 = self.xyz_to_grid(bb.p2, rounding='ceil')
        new_bb = BoundingBox(grid_pt1, grid_pt2)
        logging.debug(f'Resized BB "{bb}" to "{new_bb}"')
        return new_bb

    def precompute_node_data(self) -> None:
        """
        Precomputes node evaluation values and neighboring node info.

        :return: None
        """
        logging.debug('Precomputing node data...starting')
        p = Point3()
        for x in range(self.node_count_x):
            p.x = x
            for y in range(self.node_count_y):
                p.y = y
                for z in range(self.node_count_z):
                    p.z = z
                    node_type = self.categorize_node(p)
                    if node_type != self.NodeTypes.INVALID:
                        node_eval = self.NodeTypes.node_eval(node_type)
                        self.graph.set_node_val(deepcopy(p), node_eval)
                        logging.debug(f'Node "{p}" has eval value of {node_eval}')
                        neighbors = self._get_valid_neighbor_nodes(p)
                        for neighbor in neighbors:
                            self.graph.set_edge(deepcopy(p), deepcopy(neighbor), 1)  # initialize pheromone values to 1
        logging.debug(f'Precomputing node data...done. [{len(self.graph)} graph edges]')

    def categorize_node(self, p: Point3) -> int:
        """
        Determines the category of the specified node.
        • Invalid: node is contained within a free or occupied space
        • Superior: node exactly along the surface of an occupied space
        • General: all others

        :param p: node to test
        :return: node type id
        """
        log_msg = f'Node {p} is '
        if not self.is_in_bounds(p):
            logging.debug(log_msg + 'INVALID (out of bounds)')
            return self.NodeTypes.INVALID

        for bb in self.occupied_spaces + self.free_spaces:
            # point inside a bounding box
            if bb.contains(p, surface_inclusive=False):
                logging.debug(log_msg + 'INVALID (contained in BB)')
                return self.NodeTypes.INVALID

            # point on surface of a bounding box
            if bb.is_on_surface(p):
                # must prevent nodes from being squished between other bounding boxes and volume
                if self.volume_grid.is_on_surface(p):
                    logging.debug(log_msg + 'INVALID (squished between BB, volume)')
                    return self.NodeTypes.INVALID
                for other_bb in self.occupied_spaces + self.free_spaces:
                    if bb == other_bb:
                        continue
                    if other_bb.is_on_surface(p):
                        logging.debug(log_msg + 'INVALID (squished between two BB)')
                        return self.NodeTypes.INVALID
                logging.debug(log_msg + 'SUPERIOR')
                return Grid.NodeTypes.SUPERIOR

        logging.debug(log_msg + 'GENERAL')
        return Grid.NodeTypes.GENERAL

    def _get_valid_neighbor_nodes(self, p: Point3) -> List[Point3]:
        """
        Returns all nodes neighboring a given node that are not invalid.
        Gets orthorhombic neighbors only (i.e. excludes diagonal neighbors).

        :param p: point to inspect
        :return: list of valid neighbors (possibly empty)
        """
        candidate_points = [p + dir_ for dir_ in NEIGHBOR_DIRECTIONS]
        valid_nodes = []

        for candidate_pt in candidate_points:
            if self.categorize_node(candidate_pt) != self.NodeTypes.INVALID:
                # two neighboring points cannot be separated by a skinny BB
                for bb in self.occupied_spaces + self.free_spaces:
                    if bb.passes_through(p, candidate_pt):
                        break
                else:
                    valid_nodes.append(candidate_pt)

        return valid_nodes

    def get_neighbors_of(self, p: Point3) -> List[Point3]:
        """
        Gets all valid neighbors to the given node.

        :param p: node to inspect
        :return: list of neighbors, possibly empty
        """
        return self.graph.get_connections_to(p)

    def get_pheromone(self, p1: Point3, p2: Point3) -> float:
        """
        Gets the pheromone on the edge connecting two neighboring points.

        :param p1: first point
        :param p2: second point
        :return: pheromone value
        """
        return self.graph.get_edge(p1, p2)

    def set_pheromone(self, p1: Point3, p2: Point3, pheromone: float) -> None:
        """
        Sets the pheromone on the edge connecting two neighboring points.

        :param p1: first point
        :param p2: second point
        :param pheromone: pheromone value
        :return: pNone
        """
        self.graph.set_edge(p1, p2, pheromone)

    def evaporate_pheromones(self, rho: float) -> None:
        """
        Evaporates all pheromones in the grid given an evaporation coefficient.

        :param rho: evaporation coefficient
        :return: None
        """
        self.graph.update_edges(lambda graph_edge: graph_edge * (1 - rho))

    def is_in_bounds(self, p: Point3) -> bool:
        """
        Determines if the given grid coordinate is in bounds of the grid.

        :param p: point to inspect
        :return: True if point is in bounds, else False
        """
        return self.volume_grid.contains(p, surface_inclusive=True)

    def grid_to_xyz(self, grid_x: int, grid_y: int, grid_z: int) -> Point3:
        """
        Maps a grid coordinate to its true (x,y,z) coordinates in 3D space.

        :param grid_x: node x coord
        :param grid_y: node y coord
        :param grid_z: node z coord
        :return: point in real (x,y,z) coordinates
        """
        p = Point3()
        p.x = grid_x * self.unit_grid_size
        p.y = grid_y * self.unit_grid_size
        p.z = grid_z * self.unit_grid_size
        return p

    def xyz_to_grid(self, p: Point3, rounding: str = None) -> Point3:
        """
        Maps true (x,y,z) coordinate point to its nearest grid coordinate using
        a specified rounding method.

        :param p: true (x,y,z) coordinate point
        :param rounding: must be one of ['ceil', 'floor', 'none', None]
        :return: point representing the mapped grid node
        """
        if rounding in [None, 'none']:
            _round = lambda x: x  # noqa
        elif rounding == 'ceil':
            _round = lambda x: math.ceil(x)  # noqa
        elif rounding == 'floor':
            _round = lambda x: math.floor(x)  # noqa
        else:
            raise ValueError(f"Unknown rounding method: {rounding}. Must be one of ['ceil', 'floor', 'none', None].")

        grid_point = Point3()
        grid_point.x = _round(p.x / self.unit_grid_size)
        grid_point.y = _round(p.y / self.unit_grid_size)
        grid_point.z = _round(p.z / self.unit_grid_size)
        return grid_point.round_to_int()

    @staticmethod
    def node_list_to_elbow_list(node_list: List[Point3]) -> List[Point3]:
        """
        Converts a sequence of consecutive nodes to a list of elbows
        (inflection points). Elbow list length will always be equal to
        or shorter than the full node list. First and last nodes are
        always part of the elbow list.

        :param node_list: list of consecutive path nodes
        :return: list of nodes as elbows
        """
        assert len(node_list) >= 2, f'Node list must contain at least 2 path points; found {len(node_list)}'

        # maintain the same start position
        elbows = [node_list[0]]

        # theory: the node upstream and downstream of an elbow will have 2
        # dissimilar coordinates in 3D (e.g. different X and Z coordinates)
        for i, p in enumerate(node_list[1:-1], start=1):
            prev_node = node_list[i - 1]
            next_node = node_list[i + 1]
            dissimilarity_count = 0
            if next_node.x != prev_node.x:
                dissimilarity_count += 1
            if next_node.y != prev_node.y:
                dissimilarity_count += 1
            if next_node.z != prev_node.z:
                dissimilarity_count += 1

            if dissimilarity_count > 1:
                elbows.append(p)

        # maintain the same end position
        elbows.append(node_list[-1])
        return elbows

    @staticmethod
    def elbow_list_to_node_list(elbow_list: List[Point3]) -> List[Point3]:
        """
        Converts a sequence of elbows points to a list of consecutive path nodes.

        :param elbow_list: list of elbow points
        :return: list of consecutive nodes
        """
        assert len(elbow_list) >= 2, f'Elbow list must contain at least 2 path points; found {len(elbow_list)}'

        # maintain same start position
        nodes = [elbow_list[0]]

        for i, p in enumerate(elbow_list[1:], start=1):
            # determine direction to add next node(s)
            diff = p - elbow_list[i - 1]
            if diff.x != 0:
                diff.x = np.sign(diff.x)
            elif diff.y != 0:
                diff.y = np.sign(diff.y)
            elif diff.z != 0:
                diff.z = np.sign(diff.z)

            # keep adding diff to position until we get to the current elbow point
            loop_safety = 1e5
            loop_counter = 0
            while loop_counter < loop_safety:
                new_node = nodes[-1] + diff
                nodes.append(new_node)
                if new_node == p:
                    break
                loop_counter += 1
            else:
                raise ValueError('Infinite loop in elbow list translation')

        return nodes

    def is_elbow_path_valid(self, elbows: List[Point3]) -> bool:
        """
        Determines if every node in an elbow path is a valid node.

        :param elbows: path of elbows (inflection points)
        :return: True if path is valid, else False
        """
        node_list = self.elbow_list_to_node_list(elbows)
        for node in node_list:
            if node not in self.graph.node_values:
                return False
        return True

    def xyz_to_grid_connection(self, connection: PipeRoute) -> PipeRoute:
        """
        Converts a Connection object from XYZ coordinates to grid coordinates.
        The original connection is not modified.

        :param connection: connection to convert
        :return: new connection obj
        """
        c = deepcopy(connection)
        c.routing_start_pos = self.xyz_to_grid(c.routing_start_pos, rounding='floor')
        c.routing_end_pos = self.xyz_to_grid(c.routing_end_pos, rounding='floor')
        if c.anchor_start_pos is not None:
            c.anchor_start_pos = self.xyz_to_grid(c.anchor_start_pos, rounding='floor')
        if c.anchor_end_pos is not None:
            c.anchor_end_pos = self.xyz_to_grid(c.anchor_end_pos, rounding='floor')
        for i, point in enumerate(c.elbows):
            c.elbows[i] = self.xyz_to_grid(point)
        return c

    def normalize_pheromones(self, best_path_nodes: List[Point3]) -> None:
        self.graph.normalize_edges()

        for i in range(len(best_path_nodes) - 1):
            self.graph.set_edge(best_path_nodes[i], best_path_nodes[i + 1], 1)
