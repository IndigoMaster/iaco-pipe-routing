"""
Generic ACO solver class.
"""
import logging
from copy import deepcopy
from typing import List, Optional, Callable, Union, Tuple

import numpy as np
from tqdm import tqdm

from ant import Ant
from grid import Grid
from pipe_route import PipeRoute
from point3 import Point3
from roulette import roulette_selection
from solvers.results import SolverResult


class Solver:
    """
    A generic, hook-enabled Ant Colony Optimization solver.
    """

    class ArgMap:
        """
        Argument mapping for Solver class. Used for config file parsing.
        """
        SOLVER_NAME = 'solver_name'
        ANT_COUNT = 'ant_count'
        GRID = 'grid'
        ALPHA = 'alpha'
        BETA = 'beta'
        WEIGHT_ROUTE_LEN = 'weight_route_len'
        WEIGHT_ELBOW_COUNT = 'weight_elbow_count'
        WEIGHT_ROUTE_EVAL = 'weight_route_eval'
        ITER_COUNT = 'iterations'
        Q0 = 'q0'
        RHO = 'rho'

    def __init__(self, *,
                 solver_name: str,
                 ant_count: int,
                 grid: Grid,
                 alpha: float,
                 beta: float,
                 weight_route_len: float,
                 weight_elbow_count: float,
                 weight_route_eval: float,
                 iterations: int,
                 q0: float = 0,
                 rho: float = 0,
                 **_):
        # if arguments change, update Solver.ArgMap!
        self.solver_name: str = solver_name
        self.ant_count: int = ant_count
        self.ants: List[Ant] = [Ant(Point3.zero()) for _ in range(self.ant_count)]
        self.grid: Grid = grid
        self.iteration_count = iterations
        self.alpha = alpha
        self.beta = beta
        self.weight_route_len = weight_route_len
        self.weight_route_eval = weight_route_eval
        self.weight_elbow_count = weight_elbow_count
        self.q0 = q0
        self.rho = rho
        self.pipe_route: Union[PipeRoute, None] = None

        # solver variables (exposed for hook functions)
        self.current_iteration: int = 0
        self.successful_ants_this_iteration: List[Ant] = []

        # population fitness components
        self.global_best_fitness: float = 0
        self.global_worst_route_eval: float = 0
        self.global_worst_route_len: float = 0
        self.global_best_route_len: float = 1e9
        self.global_worst_elbow_count: int = 0
        self.min_possible_fitness = np.exp(-(self.weight_route_eval + self.weight_elbow_count + self.weight_route_len))
        self.max_possible_fitness = 1

        # hooks for solver customization
        self.hooks_enabled: bool = True
        self.hook_starting_solve: Optional[Callable[[Solver], None]] = None
        self.hook_starting_iteration: Optional[Callable[[Solver, int], None]] = None
        self.hook_starting_ant_tours: Optional[Callable[[Solver], None]] = None
        self.hook_ant_selecting_possible_moves: Optional[Callable[[Solver, Point3, List[Point3]], None]] = None
        self.hook_starting_population_fitness_calc: Optional[Callable[[Solver], None]] = None
        self.hook_starting_pheromone_evaporation: Optional[Callable[[Solver], None]] = None
        self.hook_solve_complete: Optional[Callable[[Solver], None]] = None
        self.hook_depositing_pheromones: Optional[Callable[[Solver], None]] = None
        self.custom_heuristic_func: Optional[Callable[[Solver, Point3, Point3], float]] = None

    def solve(self, route: PipeRoute, show_progress_bar: bool = True) -> SolverResult:
        """
        Solves a single instance of an Ant Colony Optimization problem
        using the ACS Algorithm.

        :param route: pipe route to solve for
        :param show_progress_bar: if True, displays CLI progress bar
        :return: results obj
        """
        self.pipe_route = route
        history_min_fitness = []
        history_mean_fitness = []
        history_max_fitness = []
        best_ant: Ant = None  # noqa

        self._call_hook(self.hook_starting_solve, [self])

        for iteration in tqdm(range(self.iteration_count), disable=(not show_progress_bar)):
            self.current_iteration = iteration
            self.successful_ants_this_iteration = []  # ants that successfully navigate to end point
            self._start_iteration(iteration)
            self._tour_ants()
            best_ant = self._calculate_fitness(history_min_fitness, history_mean_fitness, history_max_fitness, best_ant)
            self._evaporate_pheromones()
            self._deposit_pheromones(best_ant)
            logging.error(f'Solve iteration complete. '
                          f'Best fitness: {history_max_fitness[-1]:.4f} | '
                          f'Best path length: {best_ant.route_len} | '
                          f'Best elbow count: {best_ant.elbow_count}')

        logging.info('Solve procedure complete')
        self._call_hook(self.hook_solve_complete, [self])

        logging.info(f'Best route found: {best_ant.path_elbows}')
        best_route = deepcopy(self.pipe_route)
        best_route.elbows = best_ant.path_elbows
        results = SolverResult(self.solver_name, best_route)
        results.history_min_fitness = history_min_fitness
        results.history_max_fitness = history_max_fitness
        results.history_mean_fitness = history_mean_fitness
        return results

    def _start_iteration(self, iteration: int) -> None:
        """
        Solve method. Perform steps required prior to touring each ant.

        :param iteration: current iteration number
        :return: None
        """
        logging.debug(f'Beginning solver iteration {iteration}...')
        self._call_hook(self.hook_starting_iteration, [self, iteration])
        self.reset_ants()
        logging.debug(f'Ants are reset')

    def _tour_ants(self) -> None:
        """
        Solve method. Collects all logic required for touring ants.

        :return: None
        """
        self._call_hook(self.hook_starting_ant_tours, [self])
        for ant in self.ants:
            if self.tour_ant(ant, self.pipe_route.routing_end_pos):
                self.successful_ants_this_iteration.append(ant)
        good_ant_count = len(self.successful_ants_this_iteration)
        logging.info(f'{good_ant_count} ants were successful ({good_ant_count / self.ant_count:.1f}%)')

    def _calculate_fitness(self, out_history_min_fitness, out_history_mean_fitness, out_history_max_fitness, best_ant) -> Ant:
        """
        Solve method. Contains all logic required for fitness calculation.

        :param out_history_min_fitness: list of min population fitness values; modified in place
        :param out_history_mean_fitness: list of mean population fitness values; modified in place
        :param out_history_max_fitness: list of max population fitness values; modified in place
        :param best_ant: the best ant discovered so far
        :return: the best ant discovered so far (possibly unchanged)
        """
        logging.info('Computing population fitness...')
        self._call_hook(self.hook_starting_population_fitness_calc, [self])
        min_fitness, mean_fitness, max_fitness, current_best_ant = self.compute_population_fitness()
        out_history_min_fitness.append(min_fitness)
        out_history_mean_fitness.append(mean_fitness)
        out_history_max_fitness.append(max_fitness)
        if best_ant is None or current_best_ant.fitness > best_ant.fitness:
            best_ant = deepcopy(current_best_ant)
        return best_ant

    def _evaporate_pheromones(self) -> None:
        """
        Solve method. Contains logic related to pheromone evaporation.

        :return: None
        """
        self._call_hook(self.hook_starting_pheromone_evaporation, [self])
        logging.debug('Evaporating pheromones...')
        self.grid.evaporate_pheromones(self.rho)

    def _deposit_pheromones(self, best_ant: Ant) -> None:
        """
        Solve method. Contains logic related to pheromone deposition.

        :return: None
        """
        self._call_hook(self.hook_depositing_pheromones, [self])
        logging.debug('Depositing pheromones...')

        for ant in self.successful_ants_this_iteration:
            delta_pheromone = self.q0 / ant.route_len
            for i, pos1 in enumerate(ant.path_nodes[:-1]):
                pos2 = ant.path_nodes[i + 1]
                pheromone = self.grid.get_pheromone(pos1, pos2)
                pheromone += delta_pheromone
                self.grid.set_pheromone(pos1, pos2, pheromone)

        self.grid.normalize_pheromones(best_ant.path_nodes)

    def _call_hook(self, hook_func: Callable, args: List[any]) -> None:
        if self.hooks_enabled and hook_func is not None:
            logging.debug('Calling hook function...')
            hook_func(*args)
            logging.debug('Hook function complete')

    def reset_ants(self) -> None:
        """
        Resets the ant population to prepare for a new solver iteration.
        """
        for ant in self.ants:
            ant.reset_to(self.pipe_route.anchor_start_pos)
        if self.pipe_route.routing_start_pos != self.pipe_route.anchor_start_pos:
            next_node = self.pipe_route.routing_start_pos
            for ant in self.ants:
                ant.set_next_pos(next_node)

    def tour_ant(self, ant: Ant, end_pos: Point3) -> bool:
        """
        Tours a single ant through the problem space from its starting position to a target location.

        :param ant: ant to tour
        :param end_pos: end position (destination) of ant
        :return: True if ant navigated to the end position, else False
        """
        loop_safety = len(self.grid.graph)
        node_visit_count = 0
        logging.debug(f'Touring {ant}')

        while True:
            pos = ant.get_current_pos()
            logging.debug(f'Ant pos: {pos}')
            if pos == end_pos:
                logging.debug(f'Ant arrived at end position: {end_pos}')
                break

            possible_next_positions = self.grid.get_neighbors_of(pos)
            logging.debug(f'Possible next positions: {possible_next_positions}')
            logging.debug('Removing known path nodes...')
            ant.remove_visited_nodes(possible_next_positions)
            if self.pipe_route.routing_end_pos != self.pipe_route.anchor_end_pos:
                try:
                    possible_next_positions.remove(self.pipe_route.anchor_end_pos)
                except ValueError:
                    pass
            self._call_hook(self.hook_ant_selecting_possible_moves, [self, pos, possible_next_positions])
            logging.debug(f'Possible next positions: {possible_next_positions}')
            if len(possible_next_positions) == 0:
                logging.debug(f'Zero possible next positions. Navigation failed for {ant}')
                return False

            pos_probabilities = self.get_move_probabilities(pos, possible_next_positions)
            logging.debug(f'Move probabilities: {pos_probabilities}')
            next_pos = roulette_selection(pos_probabilities)
            logging.debug(f'Next position selected: {next_pos}')
            ant.set_next_pos(next_pos)

            node_visit_count += 1
            assert node_visit_count < loop_safety, 'Logic error: ant tour revisiting nodes after touring all nodes'

        if self.pipe_route.routing_end_pos != self.pipe_route.anchor_end_pos:
            ant.path_nodes.append(self.pipe_route.anchor_end_pos)
        ant.path_elbows = self.grid.node_list_to_elbow_list(ant.path_nodes)
        logging.debug(f'Navigation successful for {ant}')
        return True

    def get_move_probabilities(self, current_pos: Point3, candidate_list: List[Point3]) -> List[Tuple[Point3, float]]:
        """
        Computes the selection probabilities of each possible next position given
        a current position. Refer to Equation 7 in paper.

        :param current_pos: current position
        :param candidate_list: list of potential next positions
        :return: list of tuples; each tuple is (point obj, probability)
        """
        if len(candidate_list) == 1:
            return [(candidate_list[0], 1)]
        probs = []

        heuristic_func = self.get_current_heuristic()

        denom = 0
        for next_pos in candidate_list:
            pheromone = self.grid.get_pheromone(current_pos, next_pos)
            heuristic = heuristic_func(self, next_pos, current_pos)
            denom += (pheromone ** self.alpha) * (heuristic ** self.beta)

        for pos in candidate_list:
            pheromone = self.grid.get_pheromone(current_pos, pos)
            heuristic = heuristic_func(self, pos, current_pos)
            num = (pheromone ** self.alpha) * (heuristic ** self.beta)
            probs.append((pos, num / denom))

        return probs

    def get_current_heuristic(self) -> Callable[['Solver', Point3, Point3], float]:
        if self.custom_heuristic_func is not None:
            return self.custom_heuristic_func
        else:
            return lambda _, __, ___: 1

    def compute_population_fitness(self) -> Tuple[float, float, float, Ant]:
        """
        Computes the fitness statistics for all ants in the population.

        :return: tuple of (min_fitness, mean_fitness, max_fitness, highest_fitness_ant)
        """
        ants = self.successful_ants_this_iteration
        min_fitness = 1e9
        max_fitness = 0
        sum_fitness = 0
        best_ant = self.ants[0]

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

    def compute_ant_fitness_components(self, ant: Ant, *, update_global_trackers: bool) -> None:
        """
        Computes each individual fitness component for a given ant. Updates the ant's properties.

        :param ant: ant to inspect and update
        :param update_global_trackers: if True, updates global trackers based on this ant's fitness
        :return: None
        """
        route_eval = self.update_route_eval(ant)
        route_len = self.update_route_length(ant)
        elbow_count = ant.update_elbow_count()

        if update_global_trackers:
            # larger values are worse
            self.global_worst_route_eval = max(route_eval, self.global_worst_route_eval)
            self.global_worst_route_len = max(route_len, self.global_worst_route_len)
            self.global_best_route_len = min(route_len, self.global_best_route_len)
            self.global_worst_elbow_count = max(elbow_count, self.global_worst_elbow_count)

    def compute_combined_ant_fitness(self, ant) -> float:
        """
        Computes the fitness of a single ant. Refer to Equation 6.

        :param ant: ant to calculate and update
        :return: fitness value
        """
        w1 = self.weight_route_len
        if self.global_worst_route_len != 0:
            w1 *= ant.route_len / self.global_worst_route_len

        w2 = self.weight_elbow_count
        if self.global_worst_elbow_count != 0:
            w2 *= ant.elbow_count / self.global_worst_elbow_count

        w3 = self.weight_route_eval
        if self.global_worst_route_eval != 0:
            w3 *= ant.route_eval / self.global_worst_route_eval

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
        logging.debug(f'Route eval for {ant} is {eval_}')
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
        logging.debug(f'Route length for {ant} is {route_len}')
        return route_len

    def enable_hooks(self, enable_all: bool) -> None:
        """
        Enables or disables all hooks.

        :param enable_all: Enables all hooks if True
        :return: None
        """
        self.hooks_enabled = enable_all

    def is_valid_node(self, node: Point3) -> bool:
        """
        Determines if the given node is a valid node in the environment.

        :param node: node to test
        :return: True if node is valid
        """
        return node in self.grid.graph.edges

    def is_elbow_path_valid(self, elbows: List[Point3]) -> bool:
        """
        Determines if every node in an elbow path is a valid node.

        :param elbows: path of elbows (inflection points)
        :return: True if path is valid, else False
        """
        return self.grid.is_elbow_path_valid(elbows)
