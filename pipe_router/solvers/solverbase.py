"""
Base class for Ant Colony Optimization solvers.
"""

from typing import List, Tuple, Union

import numpy as np

from pipe_router.ant import Ant
from pipe_router.pipe_route import PipeRoute
from pipe_router.grid import Grid
from pipe_router.point3 import Point3
from pipe_router.solvers.results import SolverResults


class SolverBase:
    """The base ACO solver class all ant solvers derive from."""

    class ArgMap:
        """
        Argument mapping for SolverBase class. Used for config file parsing.
        """
        NAME = 'name'
        ANT_COUNT = 'ant_count'
        GRID = 'grid'
        ALPHA = 'alpha'
        BETA = 'beta'
        WEIGHT_ROUTE_LEN = 'weight_route_len'
        WEIGHT_ELBOW_COUNT = 'weight_elbow_count'
        WEIGHT_ROUTE_EVAL = 'weight_route_eval'
        ITER_COUNT = 'iterations'

    def __init__(self, *,
                 solver_name: str,
                 ant_count: int,
                 grid: Grid,
                 alpha: float,
                 beta: float,
                 weight_route_len: float,
                 weight_elbow_count: float,
                 weight_route_eval: float,
                 iterations: int):
        """
        Builds a solver base class with common parameters.

        :param solver_name: name of this solver
        :param ant_count: number of ants to simulate
        :param grid: grid obj
        :param alpha: alpha value
        :param beta: beta value
        :param weight_route_len: weight for route length
        :param weight_elbow_count: weight for elbow count
        :param weight_route_eval: weight for route evaluation
        :param iterations: number of iterations to simulate
        """
        self.name: str = solver_name
        self.ant_count: int = ant_count
        self.ants: List[Ant] = [Ant(Point3.zero()) for _ in range(self.ant_count)]
        self.grid: Grid = grid
        self.iteration_count = iterations
        self.alpha = alpha
        self.beta = beta
        self.weight_route_len = weight_route_len
        self.weight_route_eval = weight_route_eval
        self.weight_elbow_count = weight_elbow_count
        self.connection: Union[PipeRoute, None] = None

        # population fitness components
        self.global_best_fitness: float = 0
        self.global_worst_route_eval: float = 0
        self.global_worst_route_len: float = 0
        self.global_best_route_len: float = 1e9
        self.global_worst_elbow_count: int = 0
        self.min_possible_fitness = np.exp(-(self.weight_route_eval + self.weight_elbow_count + self.weight_route_len))
        self.max_possible_fitness = 1

    def reset_ants(self, start_pos: Point3) -> None:
        """
        Resets the ant population to prepare for a new solver iteration.
        """
        for ant in self.ants:
            ant.reset_to(start_pos)
        # add single straight section to enable route ends normal to bounding boxes
        for ant in self.ants:
            if self.connection.start_normal_dir is not None:
                next_node = self.connection.routing_start_pos
                ant.set_next_pos(next_node)

    def solve(self, conn: PipeRoute, show_progress_bar: bool = True) -> SolverResults:
        """
        Abstract method. Solves a single instance of an
        Ant Colony Optimization problem.

        :param conn: connection to solve for
        :param show_progress_bar: if True, displays CLI progress bar
        :return: results obj
        """
        raise NotImplementedError('Abstract solve method')

    @staticmethod
    def roulette_selection(pos_probs: List[Tuple[Point3, float]]) -> Point3:
        """
        Probabilistically selects a position from a list of possible positions
        where each has a selection probability.

        :param pos_probs: dictionary of {position: selection probability}
        :return: selected position
        """
        sorted_items = sorted(pos_probs, key=lambda x: x[1])  # sort by probabilities
        threshold = np.random.rand()
        cum_sum = 0
        selected: Point3 = None  # noqa
        for pos, prob in sorted_items:
            cum_sum += prob
            if threshold < cum_sum:
                return pos
        # if we didn't select any, the random threshold was large, so select the last item
        return sorted_items[-1][0]

    def compute_ant_fitness_components(self, ant: Ant, *, update_global_trackers: bool) -> None:
        """
        Computes each individual fitness component for a given ant. Updates the ant's properties.

        :param ant: ant to inspect and update
        :param update_global_trackers: if True, updates global trackers based on this ant's fitness
        :return: None
        """
        route_eval = self.update_route_eval(ant)
        route_len = self.update_route_length(ant)
        elbow_count = self.update_elbow_count(ant)

        if update_global_trackers:
            # larger values are worse
            self.global_worst_route_eval = max(route_eval, self.global_worst_route_eval)
            self.global_worst_route_len = max(route_len, self.global_worst_route_len)
            self.global_best_route_len = min(route_len, self.global_best_route_len)
            self.global_worst_elbow_count = max(elbow_count, self.global_worst_elbow_count)

    def compute_population_fitness(self, ants: List[Ant]) -> Tuple[float, float, float, Ant]:
        """
        Computes the fitness statistics for all ants in the population.

        :param ants: list of ants successful in navigation (i.e. not all ants are necessarily successful)
        :return: tuple of (min_fitness, mean_fitness, max_fitness, highest_fitness_ant)
        """
        min_fitness = 1e9
        max_fitness = 0
        sum_fitness = 0
        best_ant = None

        for ant in ants:
            self.compute_ant_fitness_components(ant, update_global_trackers=True)

        for ant in ants:
            fitness = self.compute_combined_ant_fitness(ant)
            sum_fitness += fitness
            min_fitness = min(fitness, min_fitness)
            if fitness > max_fitness:
                max_fitness = fitness
                best_ant = ant
                self.global_best_fitness = max(max_fitness, self.global_best_fitness)

        mean_fitness = sum_fitness / self.ant_count
        return min_fitness, mean_fitness, max_fitness, best_ant

    def compute_combined_ant_fitness(self, ant) -> float:
        """
        Computes the fitness of a single ant. Refer to Equation 6.

        :param ant: ant to calculate and update
        :return: fitness value
        """
        w1 = self.weight_route_len * ant.route_len / self.global_worst_route_len
        w2 = self.weight_elbow_count * ant.elbow_count / self.global_worst_elbow_count
        w3 = self.weight_route_eval * ant.route_eval / self.global_worst_route_eval
        fitness = np.exp(-(w1 + w2 + w3))
        # map to [0,1]
        scaled_fitness = (fitness - self.min_possible_fitness) / (self.max_possible_fitness - self.min_possible_fitness)
        ant.fitness = scaled_fitness
        return scaled_fitness

    def update_route_eval(self, ant: Ant) -> float:
        """
        Computes the route evaluation of a particular ant's path.
        Refer to Equations 1 and 2.

        :param ant: ant to inspect and update
        :return: eval value
        """
        eval_ = 0
        for pos in ant.path_nodes:
            eval_ += self.grid.graph.get_node_val(pos)
        ant.route_eval = eval_
        return eval_

    def update_route_length(self, ant: Ant) -> float:
        """
        Calculates the total length (distance) of an ant's path.
        Refer to Equation 3.

        :param ant: ant to inspect and update
        :return: route length
        """
        route_len = (len(ant.path_nodes) - 1) * self.grid.unit_grid_size
        ant.route_len = route_len
        return route_len

    def update_elbow_count(self, ant: Ant) -> int:
        """
        Calculates the number of elbows in an ant's path.
        Refer to Equation 4.

        :param ant: ant to inspect and update
        :return: number of elbows in path
        """
        count = len(ant.path_elbows) - 2
        ant.elbow_count = count
        return count
