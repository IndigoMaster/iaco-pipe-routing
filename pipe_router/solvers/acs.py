"""
Ant Colony Optimization solver: ACS.
"""
import logging
from copy import deepcopy
from typing import List, Tuple

from tqdm import tqdm

from pipe_route import PipeRoute
from pipe_router.ant import Ant
from pipe_router.grid import Grid
from pipe_router.point3 import Point3
from pipe_router.solvers.results import SolverResult
from pipe_router.solvers.solverbase import SolverBase
from roulette import roulette_selection


class SolverACS(SolverBase):
    """
    Ant Colony Optimization solver for original "Ant Colony Optimization" (ACO)
    intended to implement "ordinary" ACO/ACS without any IACO improvements.
    """

    class ArgMap:
        SOLVER_NAME = 'ACS'
        Q0 = 'q0'
        RHO = 'rho'

    def __init__(self, *,
                 ant_count: int,
                 grid: Grid,
                 alpha: float,
                 beta: float,
                 weight_route_len: float,
                 weight_elbow_count: float,
                 weight_route_eval: float,
                 iterations: int,
                 q0: float,
                 rho: float):
        """
        Creates an ACS solver implementing an augmented ACO algorithm.

        :param ant_count: number of ants to simulate
        :param grid: grid obj
        :param alpha: alpha value
        :param beta: beta value
        :param weight_route_len: weight for route length
        :param weight_elbow_count: weight for elbow count
        :param weight_route_eval: weight for route evaluation
        :param iterations: number of iterations to simulate
        :param q0: pheromone update constant
        :param rho: pheromone evaporation coefficient
        """
        super().__init__(solver_name=self.ArgMap.SOLVER_NAME,
                         ant_count=ant_count,
                         grid=grid,
                         alpha=alpha,
                         beta=beta,
                         weight_route_len=weight_route_len,
                         weight_elbow_count=weight_elbow_count,
                         weight_route_eval=weight_route_eval,
                         iterations=iterations)
        self.q0: float = q0
        self.rho: float = rho

    def solve(self, route: PipeRoute, show_progress_bar: bool = True) -> SolverResult:
        """
        Solves a single instance of an Ant Colony Optimization problem
        using the ACS Algorithm.

        :param route: pipe route to solve for
        :param show_progress_bar: if True, displays CLI progress bar
        :return: results obj
        """
        history_min_fitness = []
        history_mean_fitness = []
        history_max_fitness = []
        best_ant: Ant = None

        for iteration in tqdm(range(self.iteration_count)):
            logging.debug(f'Begin ACS iteration {iteration}')
            self.reset_ants()
            logging.debug(f'Ants are reset')
            successful_ants: List[Ant] = []  # ants that successfully navigate to end point

            for ant in self.ants:
                if self.tour_ant(ant, route.routing_end_pos):
                    successful_ants.append(ant)
            logging.info(f'{len(successful_ants)} ants were successful ({len(successful_ants) / self.ant_count:.1f}%)')

            logging.info('Computing population fitness...')
            min_fitness, mean_fitness, max_fitness, current_best_ant = self.compute_population_fitness(successful_ants)
            history_min_fitness.append(min_fitness)
            history_mean_fitness.append(mean_fitness)
            history_max_fitness.append(max_fitness)
            if best_ant is None or current_best_ant.fitness > best_ant.fitness:
                best_ant = deepcopy(current_best_ant)

            logging.debug('Evaporating pheromones...')
            self.grid.evaporate_pheromones(self.rho)

            logging.debug('Depositing pheromones')
            self.deposit_pheromones(successful_ants)

        logging.info(f'Best route found: {best_ant.path_elbows}')
        best_route = deepcopy(route)
        best_route.elbows = best_ant.path_elbows
        results = SolverResult(self.ArgMap.SOLVER_NAME, best_route)
        results.history_min_fitness = history_min_fitness
        results.history_max_fitness = history_max_fitness
        results.history_mean_fitness = history_mean_fitness
        return results

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
        probs = []
        denom = 0
        for next_pos in candidate_list:
            pheromone = self.grid.get_pheromone(current_pos, next_pos)
            dist = 1  # constant due to univariate grid size
            denom += (pheromone ** self.alpha) * (dist ** self.beta)

        for pos in candidate_list:
            pheromone = self.grid.get_pheromone(current_pos, pos)
            dist = 1  # constant due to univariate grid size
            num = (pheromone ** self.alpha) * (dist ** self.beta)
            probs.append((pos, num / denom))

        return probs

    def deposit_pheromones(self, successful_ants: List[Ant]) -> None:
        """
        Updates the pheromone graph using routes from all successful ants.
        Refer to Equations 9 and 10 in paper.

        :param successful_ants: list of ants that successfully navigated to the end point
        :return: None
        """
        for ant in successful_ants:
            delta_pheromone = self.q0 / ant.route_len
            for i, pos1 in enumerate(ant.path_nodes[:-1]):
                pos2 = ant.path_nodes[i + 1]
                pheromone = self.grid.get_pheromone(pos1, pos2)
                pheromone += delta_pheromone
                self.grid.set_pheromone(pos1, pos2, pheromone)
