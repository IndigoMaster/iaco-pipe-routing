"""
Solver Results data class.
"""

import numpy as np

from pipe_router.pipe_route import PipeRoute


class SolverResult:
    """
    Data class for organizing results of an Ant Colony Optimization solver.
    """

    def __init__(self, solver_name: str, best_pipe_route: PipeRoute):
        """
        Creates a new results object.

        :param solver_name: name of solver responsible for these results (e.g. 'ACS')
        :param best_pipe_route: best pipe route identified
        """
        self.solver_name = solver_name
        self.best_pipe_route: PipeRoute = best_pipe_route
        self.history_min_fitness: np.ndarray = np.array([])
        self.history_mean_fitness: np.ndarray = np.array([])
        self.history_max_fitness: np.ndarray = np.array([])
