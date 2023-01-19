"""
Base class for specific solver strategies.
"""

from pipe_route import PipeRoute
from solvers.results import SolverResult
from solvers.solver import Solver


class SolverBase:
    """
    Abstract base class for a solver strategy.
    """

    class ArgMap:
        """
        Argument mapping for ACS solver class. Used for config file parsing.
        """
        SOLVER_NAME_VAR = 'name'
        SOLVER_NAME = '__base__'

    def __init__(self, **kwargs):
        if 'solver_name' in kwargs:
            self.solver_name = kwargs.pop('solver_name')
        else:
            self.solver_name = self.ArgMap.SOLVER_NAME
        self.solver = Solver(solver_name=self.solver_name, **kwargs)

    def solve(self, route: PipeRoute, show_progress_bar: bool = True) -> SolverResult:
        """
        Abstract method. Runs solve operation on the configured Solver object.

        :param route: pipe route to solve for
        :param show_progress_bar: if True, displays progress
        :return: solver result object
        """
        raise NotImplementedError('Abstract base class')
