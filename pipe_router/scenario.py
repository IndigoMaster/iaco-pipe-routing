"""
Pipe routing problem class
"""
import logging
from typing import List

from tqdm import tqdm

from pipe_router.bounding_box import BoundingBox
from pipe_router.grid import Grid
from pipe_router.parser import Params
from pipe_router.pipe_route import PipeRoute
from solvers._solverbase import SolverBase
from solvers.results import SolverResult


class PipeRoutingScenario:
    """
    Represents a single pipe routing problem.
    """

    def __init__(self, params: Params):
        """
        Creates a new pipe routing problem instance.

        :param params: populated Params object
        """
        logging.info('Constructing new scenario...')
        self.params: Params = params
        self.volume: BoundingBox = params.volume
        self.grid: Grid = params.grid
        self.occupied_space_boxes: List[BoundingBox] = params.occupied_space_boxes
        self.free_space_boxes: List[BoundingBox] = params.free_space_boxes
        self.pipe_routes: List[PipeRoute] = params.pipe_routes
        self.solvers: List[SolverBase] = params.solver_list
        self._setup_problem()

    def solve(self) -> List[SolverResult]:
        """
        Solves a Pipe Routing Problem, which may include many sub-problems.
        Each sub-problem is solved sequentially. Solving order is not
        guaranteed.

        :return: list of results objects
        """
        results = []

        if len(self.pipe_routes) == 0:
            logging.info("No pipe routes defined. Skipping solve procedure.")
            return []

        if len(self.solvers) == 0:
            logging.info("No solvers defined. Skipping solve procedure.")
            return []

        for pipe_route in tqdm(self.pipe_routes, desc='Pipe Routes', leave=False, disable=self.params.quiet):
            for solver in tqdm(self.solvers, desc='Solvers    ', leave=False, disable=self.params.quiet):
                logging.info(f'Starting solver "{solver.solver_name}" for pipe route "{pipe_route.name}"')
                result = solver.solve(route=pipe_route, show_progress_bar=(not self.params.quiet))
                results.append(result)
        return results

    def _setup_problem(self) -> None:
        """
        Configures problem elements.

        :return: None
        """
        self._setup_grid()
        self._setup_pipe_routes()

    def _setup_grid(self) -> None:
        """
        Configures the Grid object for this problem.

        :return: None
        """
        for bb in self.occupied_space_boxes:
            logging.debug(f'Adding occupied BB to grid: "{bb.name}"')
            self.grid.add_occupied_box(bb)
        for bb in self.free_space_boxes:
            logging.debug(f'Adding free BB to grid: "{bb.name}"')
            self.grid.add_free_space_box(bb)
        self.grid.precompute_node_data()

    def _setup_pipe_routes(self) -> None:
        """
        Configures the pipe routes for this problem.

        :return: None
        """
        for i, pipe_route in enumerate(self.pipe_routes):
            logging.debug(f'Configuring pipe route "{pipe_route.name}..."')
            pipe_route_grid_coords = self.grid.xyz_to_grid_connection(pipe_route)

            # if pipe route is associated with a bounding box, establish the direction normal to the surface
            start_normal_dir = None
            end_normal_dir = None
            for bb in self.grid.occupied_spaces:  # use grid bb's because they are mapped to grid coords
                if bb.contains(pipe_route_grid_coords.routing_start_pos, surface_inclusive=True):
                    normal_dir = bb.get_normal_direction(pipe_route_grid_coords.routing_start_pos)
                    start_normal_dir = normal_dir
                    break
            for bb in self.grid.occupied_spaces:
                if bb.contains(pipe_route_grid_coords.routing_end_pos, surface_inclusive=True):
                    normal_dir = bb.get_normal_direction(pipe_route_grid_coords.routing_end_pos)
                    end_normal_dir = normal_dir
                    break
            pipe_route_grid_coords.update_terminals_from_normals(start_normal_dir, end_normal_dir, self.grid.unit_grid_size)

            self.pipe_routes[i] = pipe_route_grid_coords
