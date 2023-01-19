"""
Ant Colony Optimization solver: IACO strategy.
"""
import logging
from copy import deepcopy
from typing import List

import numpy as np

from pipe_router.ant import Ant
from pipe_router.grid import Grid
from pipe_router.pipe_route import PipeRoute
from pipe_router.point3 import Point3
from solvers.bacs import SolverBACS
from solvers.solver import Solver

SUBPATH_MUTATION_ANT_COUNT = 5
SUBPATH_MUTATION_ITERATION_COUNT = 8


class SolverIACO(SolverBACS):
    """
    IACO solver strategy implementation.
    """

    class ArgMap(SolverBACS.ArgMap):
        """
        Argument mapping for IACO solver class. Used for config file parsing.
        """
        SOLVER_NAME = 'IACO'

    def __init__(self, **kwargs):
        # if argument names change, update Solver.ArgMap!
        if 'solver_name' in kwargs:
            self.solver_name = kwargs.pop('solver_name')
        else:
            self.solver_name = self.ArgMap.SOLVER_NAME
        super().__init__(solver_name=self.solver_name, **kwargs)
        self.configure_solver()
        self.subpath_solver: Solver = self.build_subpath_solver(**kwargs)

    def configure_solver(self) -> None:
        """
        Configures the solver instance with IACO functionality.

        :return: None
        """
        # IACO is the same as BACS with the addition of the heuristic function and mutation
        # mechanism. Refer to section 4 in paper.
        self.solver.hook_starting_population_fitness_calc = self.mutate_population_hook
        self.solver.custom_heuristic_func = self.iaco_heuristic_func

    def iaco_heuristic_func(self, solver_instance: Solver, current_pos: Point3, prev_pos: Point3) -> float:
        """
        Solver function adapter. Enables solver to calculate the IACO heuristic.

        :param solver_instance:
        :param current_pos:
        :param prev_pos:
        :return:
        """
        return self.calc_heuristic(start_pos=solver_instance.pipe_route.anchor_start_pos,
                                   end_pos=solver_instance.pipe_route.anchor_end_pos,
                                   current_pos=current_pos,
                                   prev_pos=prev_pos)

    @staticmethod
    def calc_heuristic(start_pos: Point3, end_pos: Point3, current_pos: Point3, prev_pos: Point3) -> float:
        """
        Calculates the "modified heuristic function" that takes into account the distance
        to the end position and angle of most recent move vs. ideal angle of travel.
        Refer to Equation 12 in paper.

        :param start_pos: connection start position
        :param end_pos: connection end position
        :param current_pos: current node position
        :param prev_pos: previous node position
        :return: heuristic value
        """
        num = (current_pos.x - prev_pos.x) * (end_pos.x - start_pos.x) + \
              (current_pos.y - prev_pos.y) * (end_pos.y - start_pos.y) + \
              (current_pos.z - prev_pos.z) * (end_pos.z - start_pos.z)
        denom = np.sqrt((current_pos.x - prev_pos.x) ** 2 +
                        (current_pos.y - prev_pos.y) ** 2 +
                        (current_pos.z - prev_pos.z) ** 2) * \
                np.sqrt((end_pos.x - start_pos.x) ** 2 +
                        (end_pos.y - start_pos.y) ** 2 +
                        (end_pos.z - start_pos.z) ** 2)
        angle = np.arccos(num / denom)
        # equation 11
        dist_to_end_pos = np.sqrt((current_pos.x - end_pos.x) ** 2 +
                                  (current_pos.y - end_pos.y) ** 2 +
                                  (current_pos.z - end_pos.z) ** 2)
        if angle < np.pi:
            heuristic = (2 / dist_to_end_pos) if (dist_to_end_pos != 0) else 2
        else:
            heuristic = (1 / dist_to_end_pos) if (dist_to_end_pos != 0) else 1
        return heuristic

    def mutate_population_hook(self, solver_instance: Solver) -> None:
        """
        Solver hook function. Performs mutation operation on population.
        Solver is modified in-place.

        :param solver_instance: solver instance
        :return: None
        """
        logging.debug('Entered hook: mutate population')
        solver_instance.enable_hooks(False)  # prevent hook recursion
        solver_instance.compute_population_fitness()
        self.mutate_population(solver_instance)
        solver_instance.enable_hooks(True)

    def mutate_population(self, solver_instance: Solver) -> None:
        """
        Performs mutation operation on the (successful) ant population.

        :return: None (ants modified in place)
        """
        for ant in solver_instance.successful_ants_this_iteration:
            if np.random.rand() < 0.5: return
            logging.debug(f'Mutating {ant}...')
            # self.do_subpath_mutation(ant, solver_instance)
            self.do_elbow_mutation(ant, solver_instance)

    def do_subpath_mutation(self, ant: Ant, solver_instance: Solver) -> None:
        """
        Constructs a new path between two randomly selected path nodes.

        :return: None (ant is modified in-place)
        """
        logging.debug(f'Subpath mutation begin')
        subpath_start_index = np.random.randint(0, len(ant.path_nodes) - 1)
        subpath_end_index = np.random.randint(subpath_start_index + 1, len(ant.path_nodes))
        start_pos = ant.path_nodes[subpath_start_index]
        end_pos = ant.path_nodes[subpath_end_index]
        existing_path_nodes = ant.path_nodes[0:subpath_start_index] + ant.path_nodes[subpath_end_index + 1:]
        logging.debug(f'Finding subpath from index {subpath_start_index} to {subpath_end_index}')

        def eliminate_existing_node_options_hook(_: Solver, __: Point3, possible_next_moves: List[Point3]) -> None:
            """
            Removes all nodes that belong to the original ant's existing path to prevent visiting
            nodes multiple times.

            :param _: solver instance
            :param __: current position
            :param possible_next_moves: list of next move candidate positions
            :return: None (list modified in place)
            """
            for node in existing_path_nodes:
                if node in possible_next_moves:
                    possible_next_moves.remove(node)

        self.subpath_solver.hook_ant_selecting_possible_moves = eliminate_existing_node_options_hook
        logging.debug('Hook established')
        subpath_pipe_route = PipeRoute(start_point=start_pos, end_point=end_pos)
        result = self.subpath_solver.solve(subpath_pipe_route, False)
        if result.best_pipe_route is None:
            logging.debug('No valid subpath found')
            return
        else:
            logging.debug('Valid subpath found')

        mutated_ant = deepcopy(ant)
        mutated_ant.path_elbows = result.best_pipe_route.elbows
        mutated_fitness = solver_instance.compute_combined_ant_fitness(mutated_ant)
        logging.debug(f'Original fitness: {ant.fitness} | New fitness: {mutated_fitness}')

        if mutated_fitness > ant.fitness:
            logging.debug('Subpath mutation success')
            ant.path_nodes = mutated_ant.path_nodes
            ant.path_elbows = mutated_ant.path_elbows
            ant.fitness = mutated_fitness
        else:
            logging.debug('Subpath mutation failure')

    def do_elbow_mutation(self, ant: Ant, solver_instance: Solver) -> None:
        """
        Selects three consecutive inflection points (three elbows) and determines if the
        reversed path between them is viable. The highest-fitness option is kept.

        :param ant: ant to perform mutation operation on
        :param solver_instance: solver instance
        :return: None; ant is modified in place
        """
        logging.debug(f'Elbow mutation begin')

        # randomly select first of 3 consecutive elbows
        start_index = np.random.randint(0, len(ant.path_elbows) - 2)
        elbow_index = start_index + 1
        logging.debug(f'Selected elbow sequence is {ant.path_elbows[start_index:start_index + 3]}')
        mutated_elbow = self.get_fictitious_elbow(*ant.path_elbows[start_index:start_index + 3])
        logging.debug(f'Mutated elbow: {mutated_elbow}')

        # if fictitious node is invalid or already in path, ignore this attempt
        if (not solver_instance.is_valid_node(mutated_elbow)) or (mutated_elbow in ant.path_nodes):
            logging.debug('Elbow mutation failure: invalid elbow')
            return

        # verify validity of overall path (intermediate nodes might not be valid)
        mutated_ant = deepcopy(ant)
        mutated_ant.path_elbows[elbow_index] = mutated_elbow
        self.remove_unnecessary_elbows(mutated_ant)  # simplify mutated path
        if not solver_instance.is_elbow_path_valid(mutated_ant.path_elbows):
            logging.debug('Elbow mutation failure: invalid path')
            return

        # keep mutated path if it is higher fitness than original
        mutated_fitness = solver_instance.compute_combined_ant_fitness(mutated_ant)
        logging.debug(f'Original fitness: {ant.fitness} | New fitness: {mutated_fitness}')
        if mutated_fitness > ant.fitness:
            logging.error(f'Elbow mutation success: Original fitness: {ant.fitness} | New fitness: {mutated_fitness}')
            ant.path_nodes = mutated_ant.path_nodes
            ant.path_elbows[elbow_index] = mutated_elbow
            ant.fitness = mutated_fitness
        else:
            logging.debug('Elbow mutation failure: insufficient fitness')

    @staticmethod
    def get_fictitious_elbow(p1: Point3, p2: Point3, p3: Point3) -> Point3:
        """
        Determines the location of the fictitious elbow that completes the rectangle
        defined by three corner points.

        :param p1: first elbow
        :param p2: second elbow
        :param p3: third elbow
        :return: fictitious elbow
        """
        elbow = deepcopy(p2)
        if p1.x == p2.x == p3.x:
            if p1.y == p2.y:
                elbow.y = p3.y
                elbow.z = p1.z
            else:
                elbow.y = p1.y
                elbow.z = p3.z
        elif p1.y == p2.y == p3.y:
            if p1.x == p2.x:
                elbow.x = p3.x
                elbow.z = p1.z
            else:
                elbow.x = p1.x
                elbow.z = p3.z
        elif p1.z == p2.z == p3.z:
            if p1.x == p2.x:
                elbow.x = p3.x
                elbow.y = p1.y
            else:
                elbow.x = p1.x
                elbow.y = p3.y
        return elbow

    @staticmethod
    def remove_unnecessary_elbows(ant: Ant) -> None:
        """
        Removes unnecessary elbows (inflection points) from an ant's path.
        Updates both path nodes and inflection nodes in the ant.

        :param ant: ant with path to inspect
        :return: None (ant is modified in place)
        """
        node_path = Grid.elbow_list_to_node_list(ant.path_elbows)
        new_inflection_path = Grid.node_list_to_elbow_list(node_path)
        assert len(node_path) == len(ant.path_nodes), 'Elbow simplification should not change the number of nodes in a path'
        assert len(new_inflection_path) <= len(ant.path_elbows), 'Elbow simplification should never increase the number of elbows in a path'
        ant.path_nodes = node_path
        ant.path_elbows = new_inflection_path
        ant.update_elbow_count()

    def build_subpath_solver(self, **kwargs) -> Solver:
        """
        Constructs a solver to handle all subpath mutations.

        :param kwargs: kwargs passed into main solver
        :return: constructed solver
        """
        _kwargs = deepcopy(kwargs)
        _kwargs['ant_count'] = SUBPATH_MUTATION_ANT_COUNT
        _kwargs['iterations'] = SUBPATH_MUTATION_ITERATION_COUNT
        subpath_solver = Solver(solver_name='subpath solver', **_kwargs)
        subpath_solver.grid = self.solver.grid
        return subpath_solver
