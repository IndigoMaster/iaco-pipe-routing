"""
Ant Colony Optimization solver: BACS strategy.
"""
from typing import List, Callable

import numpy as np

from point3 import Point3
from solvers.acs import SolverACS
from solvers.solver import Solver


class SolverBACS(SolverACS):
    """
    BACS solver strategy implementation.
    """

    class ArgMap:
        """
        Argument mapping for BACS solver class. Used for config file parsing.
        """
        SOLVER_NAME = 'BACS'
        Q_MIN = 'q_min'
        Q_MAX = 'q_max'
        RHO_MIN = 'rho_min'
        RHO_MAX = 'rho_max'

    def __init__(self, **kwargs):
        # if argument names change, update Solver.ArgMap!
        if 'solver_name' in kwargs:
            self.solver_name = kwargs.pop('solver_name')
        else:
            self.solver_name = self.ArgMap.SOLVER_NAME
        self.q_min: float = kwargs.pop('q_min')
        self.q_max: float = kwargs.pop('q_max')
        self.rho_min: float = kwargs.pop('rho_min')
        self.rho_max: float = kwargs.pop('rho_max')
        super().__init__(solver_name=self.solver_name, **kwargs)
        self.configure_solver()

    def configure_solver(self) -> None:
        """
        Configures the solver instance with BACS functionality.

        :return: None
        """
        # BACS is the same as ACS with the addition of the dynamic parameter
        # mechanism. Refer to section 4 in paper.
        self.solver.hook_starting_solve = self.get_hook__update_dynamic_params()  # initialize dynamic params
        self.solver.hook_starting_pheromone_evaporation = self.get_hook__update_dynamic_params()
        self.solver.hook_ant_selecting_possible_moves = self.possibly_choose_pheromone_based_move

    def get_hook__update_dynamic_params(self) -> Callable[[Solver], None]:
        """
        Implements the "dynamical parameter mechanism" of IACO, where
        q0 and rho parameters are adapted to current conditions.
        Linearly decreases q0 and rho from their max values to min values
        over the course of the simulation. Refer to Equations 16 and 17 in paper.

        :return: hook function
        """

        def _hook(solver_instance: Solver) -> None:
            current_iteration = solver_instance.current_iteration
            total_iterations = solver_instance.iteration_count
            solver_instance.q0 = self.q_max - (self.q_max - self.q_min) * current_iteration / total_iterations
            solver_instance.rho = self.rho_max - (self.rho_max - self.rho_min) * current_iteration / total_iterations

        return _hook

    @staticmethod
    def possibly_choose_pheromone_based_move(solver_instance: Solver, current_pos: Point3, possible_next_moves: List[Point3]) -> List[Point3]:
        """
        Solver hook function. Potentially chooses to select the next move based
        on the threshold q0. If this happens, a single move will be chosen if any
        valid moves exist, based only on pheromone levels. Otherwise, do nothing.
        Implements Equation 13 in paper.

        :param solver_instance: solver instance
        :param current_pos: current ant position
        :param possible_next_moves: list of candidate positions for next move
        :return: list of moves (possibly of length 1 if pheromone based move used)
        """
        if np.random.random() <= solver_instance.q0:
            if len(possible_next_moves) == 0:
                return []
            else:
                chosen_move = SolverBACS.choose_pheromone_based_move(solver_instance, current_pos, possible_next_moves)
                return [chosen_move]
        else:
            return possible_next_moves

    @staticmethod
    def choose_pheromone_based_move(solver_instance: Solver, current_pos: Point3, candidate_list: List[Point3]) -> Point3:
        """
        Selects the next move from a list of candidate moves based solely on
        pheromone and heuristic information. Implements Equation 13 in paper.

        :param solver_instance: solver instance
        :param current_pos: current position
        :param candidate_list: list of candidate positions for next move
        :return: selected move
        """
        # equation 13
        max_pheromone = -1e9
        best_next_pos_list = []  # multiple paths may have identical pheromones; randomly select among them
        for next_pos in candidate_list:
            pheromone = solver_instance.grid.get_pheromone(current_pos, next_pos)
            heuristic = solver_instance.get_current_heuristic()(solver_instance, next_pos, current_pos)
            val = (pheromone ** solver_instance.alpha) * (heuristic ** solver_instance.beta)
            if val == max_pheromone:
                best_next_pos_list.append(next_pos)
            elif val > max_pheromone:
                max_pheromone = val
                best_next_pos_list = [next_pos]
        selected = np.random.choice(best_next_pos_list)
        return selected
