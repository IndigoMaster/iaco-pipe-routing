"""
Ant Colony Optimization solver: ACS strategy.
"""

from pipe_route import PipeRoute
from pipe_router.solvers.results import SolverResult
from solvers.solver_base import SolverBase


class SolverACS(SolverBase):
    """
    ACS solver strategy implementation.
    """

    class ArgMap:
        """
        Argument mapping for ACS solver class. Used for config file parsing.
        """
        SOLVER_NAME = 'ACS'
        Q0 = 'q0'
        RHO = 'rho'

    def __init__(self, **kwargs):
        if 'solver_name' in kwargs:
            self.solver_name = kwargs.pop('solver_name')
        else:
            self.solver_name = self.ArgMap.SOLVER_NAME
        super().__init__(solver_name=self.solver_name, **kwargs)

    def solve(self, route: PipeRoute, show_progress_bar: bool = True) -> SolverResult:
        """
        Runs ACS solver on a single pipe route problem.

        :param route: pipe route to solve for
        :param show_progress_bar: if True, displays progress
        :return: solver result object
        """
        return self.solver.solve(route, show_progress_bar)
