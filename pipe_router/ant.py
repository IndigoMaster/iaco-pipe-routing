"""
Ant class definition.
"""
from typing import List

import numpy as np

from pipe_router.point3 import Point3


class Ant:
    """
    Data class representing a single Ant for Ant Colony Optimization algorithms.
    """

    def __init__(self, start_point: Point3):
        """
        Constructs an empty Ant.
        """
        self.path_nodes: List[Point3] = [start_point]  # originating node is index 0
        self.path_elbows: List[Point3] = [start_point]  # originating node is index 0
        self.elbow_count: int = 0
        self.route_eval: float = 0
        self.route_len: float = 0
        self.fitness: float = 0
        self.fitness_values_valid: bool = False
        self.id = ''.join(list(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5))).upper()

    def reset_to(self, pos: Point3) -> None:
        """
        Resets the ant position to a new position and starts a new path.

        :param pos: new starting position
        :return: None
        """
        self.__init__(pos)

    def get_current_pos(self) -> Point3:
        """
        Gets the current position of this Ant.

        :return: point as position
        """
        return self.path_nodes[-1]

    def set_next_pos(self, new_pos: Point3, throw: bool = True) -> bool:
        """
        Assigns the ant to a new position and updates the ant's path with
        the new position. Optionally throws an exception if the new position
        is already part of the current path.

        :param new_pos: new position for ant
        :param throw: if True, raises an exception if the new position
            is already part of the current path
        :return: True if this new position is valid (not yet visited), else False
        """
        already_visited = new_pos in self.path_nodes
        if already_visited and throw:
            raise ValueError(f'Attempted to assign ant to a previously visited node: {new_pos}')
        self.path_nodes.append(new_pos)
        return already_visited

    def remove_visited_nodes(self, out_candidate_list: List[Point3]) -> None:
        """
        Modifies in-place a list of nodes to remove previously visited
        nodes.

        :param out_candidate_list: list of candidate positions to be modified in-place
        :return: None; list modified in-place
        """
        invalid_points = self.path_nodes
        for i in range(len(out_candidate_list) - 1, -1, -1):  # iterate backwards
            if out_candidate_list[i] in invalid_points:
                out_candidate_list.pop(i)

    def update_elbow_count(self) -> int:
        """
        Calculates the number of elbows in the current path.
        Refer to Equation 4.

        :return: number of elbows in path
        """
        count = len(self.path_elbows) - 2
        self.elbow_count = count
        return count

    def __str__(self):
        return f'Ant {self.id}: @{self.get_current_pos()}; node_count={len(self.path_nodes)}'

    def __repr__(self):
        return self.__str__()
