"""
Params object definition.
"""
from pathlib import Path
from typing import List, Optional

from pipe_router.bounding_box import BoundingBox
from pipe_router.grid import Grid
from pipe_router.pipe_route import PipeRoute
from pipe_router.solvers.solverbase import SolverBase
from point3 import Point3


class Params:
    """
    Data class for storing parameters read from a config file.
    """

    def __init__(self):
        """
        Constructs an empty Params object
        """
        self.random_seed: float = -1
        self.config_file_path: Path = Path()
        self.quiet: bool = False
        self.output_dir_path: Path = Path()

        self.scenario_title: str = "No Title"
        self.volume: Optional[BoundingBox] = None
        self.grid: Optional[Grid] = None
        self.occupied_space_boxes: List[BoundingBox] = []
        self.free_space_boxes: List[BoundingBox] = []
        self.pipe_routes: List[PipeRoute] = []
        self.solver_list: List[SolverBase] = []

    def set_volume(self, volume_size: Point3) -> None:
        """
        Sets the volume for these params.

        :param volume_size: volume bounding box size
        """
        self.volume = BoundingBox(Point3.zero(), volume_size)

    def set_grid(self, grid: Grid) -> None:
        """
        Sets the grid object for these params.

        :param grid: grid object
        """
        self.grid = grid

    def add_occupied_space_box(self, bb: BoundingBox) -> None:
        """
        Appends a new occupied space bounding box to the collection.

        :param bb: bounding box representing an occupied space
        """
        self.occupied_space_boxes.append(bb)

    def add_free_space_box(self, bb: BoundingBox) -> None:
        """
        Appends a new free space bounding box to the collection.

        :param bb: bounding box representing a free space
        """
        self.free_space_boxes.append(bb)

    def add_pipe_route(self, route: PipeRoute) -> None:
        """
        Appends a new pipe route object to the collection.

        :param route: pipe route object
        """
        self.pipe_routes.append(route)

    def add_solver(self, solver: SolverBase) -> None:
        """
        Appends a new solver object to the collection.

        :param solver: solver obj
        """
        self.solver_list.append(solver)

    def set_scenario_title(self, title: str) -> None:
        """
        Sets the current title.

        :param title: title string
        """
        self.scenario_title = title
