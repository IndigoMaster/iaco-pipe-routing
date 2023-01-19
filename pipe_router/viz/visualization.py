"""
IACO problem visualization class.
"""
from copy import deepcopy
from typing import List, Dict, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# https://www.sharetechnote.com/html/Octave_Matlab_3DGraph.html
from pipe_router.bounding_box import BoundingBox
from pipe_router.grid import Grid
from pipe_router.pipe_route import PipeRoute
from pipe_router.point3 import Point3
from pipe_router.scenario import PipeRoutingScenario
from pipe_router.solvers.results import SolverResult
from pipe_router.viz.drawing_constants import *


class Visualizer:
    """
    Utility class for visualizing a pipe routing problem environment.
    """

    def __init__(self):
        self.volume: Union[BoundingBox, None] = None
        self.grid: Union[Grid, None] = None
        self.occupied_spaces: List[BoundingBox] = []
        self.free_spaces: List[BoundingBox] = []
        self.solver_results: List[SolverResult] = []
        self.fig: Union[plt.Figure, None] = None
        self.ax: Union[plt.Axes, None] = None
        self.show_grid: bool = False

    @staticmethod
    def from_scenario(scenario: PipeRoutingScenario, results: List[SolverResult]) -> 'Visualizer':
        """
        Builds and populates a visualizer object given a fully solved scenario.

        :param scenario: solved scenario to visualize
        :param results: solver results
        :return: populated visualizer object
        """
        viz = Visualizer()
        viz.set_volume(scenario.volume)
        viz.set_grid(scenario.grid)
        viz.add_occupied_spaces(scenario.occupied_space_boxes)
        viz.add_free_spaces(scenario.free_space_boxes)
        viz.solver_results = results
        return viz

    def set_volume(self, volume: BoundingBox) -> None:
        """
        Sets the volume. Overwrites the existing volume, if any.

        :param volume: bounding box representing the working volume.
        :return: None
        """
        self.volume = volume

    def set_grid(self, grid: Grid) -> None:
        """
        Sets the Grid. Overwrites the existing Grid object, if any.

        :param grid: grid obj
        :return: None
        """
        self.grid = grid

    def add_occupied_space(self, bb: BoundingBox) -> None:
        """
        Adds an occupied space the current collection.

        :param bb: bounding box representing an occupied space
        :return: None
        """
        self.occupied_spaces.append(bb)

    def add_occupied_spaces(self, bb_list: List[BoundingBox]) -> None:
        """
        Adds multiple occupied spaces to the current collection.

        :param bb_list: list of bounding boxes to add
        :return: None
        """
        self.occupied_spaces.extend(bb_list)

    def add_free_space(self, bb: BoundingBox) -> None:
        """
        Adds a free space to the current collection.

        :param bb: bounding box representing a free space
        :return: None
        """
        self.free_spaces.append(bb)

    def add_free_spaces(self, bb_list: List[BoundingBox]) -> None:
        """
        Adds multiple free spaces to the current collection.

        :param bb_list: list of bounding boxes to add
        :return: None
        """
        self.free_spaces.extend(bb_list)

    def add_pipe_routes(self, routes: List[PipeRoute]) -> None:
        """
        Adds multiple pipe routes to the current collection.

        :param routes: list of pipe route objects to add
        :return: None
        """
        self.pipe_routes.extend(routes)

    def render(self, *,
               show: bool = False,
               title: Optional[str] = None,
               show_axes: bool = False,
               show_grid: bool = False,
               hold_on: bool = False) -> plt.Figure:
        """
        Generates an interactive 3D plot according to the current configuration.

        :param show: if True, calls plt.show() after plotting is complete (blocking call)
        :param title: title of plot; optional
        :param show_axes: if True, shows axis labels and tick marks. Default is False.
        :param show_grid: if True, shows grid. Default is False.
        :param hold_on: if True and a previously built plot exists, then draws (appends)
            to the current plot instead of constructing a new one. Default is False.
        :return: handle to figure
        """
        if not hold_on or self.fig is None or self.ax is None:
            self._build_plot()
        self._populate_plot()

        if title is not None:
            self.ax.set_title(title)

        if show_axes:
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
        else:
            self.ax.axis('off')

        self.show_grid = show_grid

        self.ax.grid(visible=False)
        self.ax.set_box_aspect([1, 1, 1])  # noqa
        plt.tight_layout()

        if show:
            plt.show()

        return self.fig

    def _build_plot(self) -> None:
        """
        Constructs a 3D plot figure handle and Axes object. If an open plot
        is currently associated with this instance, the plot will be closed
        prior to constructing a new plot.

        :return: None
        """
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_xticks(np.linspace(0, self.grid.true_3d_size.x, self.grid.node_count_x))
        self.ax.set_yticks(np.linspace(0, self.grid.true_3d_size.y, self.grid.node_count_y))
        self.ax.set_zticks(np.linspace(0, self.grid.true_3d_size.z, self.grid.node_count_z))

    def _populate_plot(self) -> None:
        """
        Populates the existing plot with objects (volume, grid, boxes, pipe routes).

        :return: None
        """
        pipe_routes_present = False

        # draw volume
        if self.volume is not None:
            self._draw_box(self.volume, DRAW_ARGS_VOLUME)

        # draw grid cells
        if self.grid is not None and self.show_grid:
            self._draw_grid()

        # draw pipe routes
        if self.solver_results is not None and len(self.solver_results) > 0:
            pipe_routes_present = True
            self._draw_pipe_routes(DRAW_ARGS_PIPE_ROUTE)

        # draw occupied spaces
        if self.occupied_spaces is not None and len(self.occupied_spaces) > 0:
            self._draw_boxes(self.occupied_spaces, DRAW_ARGS_OCCUPIED_SPACE)

        # draw free spaces
        if self.free_spaces is not None and len(self.free_spaces) > 0:
            self._draw_boxes(self.free_spaces, DRAW_ARGS_FREE_SPACE)

        # legend
        if pipe_routes_present:
            self._draw_legend()

    def _draw_boxes(self, bb_list: List[BoundingBox], draw_args: Dict) -> None:
        """
        Draws many boxes using the same drawing args.

        :param bb_list: list of bounding boxes to draw
        :param draw_args: drawing args to use for all bounding boxes
        :return: None
        """
        for bb in bb_list:
            self._draw_box(bb, draw_args)

    def _draw_box(self, bb: BoundingBox, draw_args: Dict) -> None:
        """
        Draws a generic box:
            To draw a shaded box, draw_args must include '_box_type': 'shaded'.
            To draw a wireframe box, draw_args must include '_box_type': 'wireframe'.
            To draw a shaded wireframe box, draw_args must include '_box_type': 'shaded_wireframe'.
        Defaults to a wireframe box.

        :param bb: bounding box representing the box to draw
        :param draw_args: drawing args
        :return: None
        """
        draw_args = deepcopy(draw_args)
        if '_box_type' not in draw_args:
            draw_args['_box_type'] = 'wireframe'

        box_type = draw_args['_box_type']
        draw_args.pop('_box_type')
        if box_type == 'shaded':
            self._draw_box_shaded(bb, draw_args)
        elif box_type == 'wireframe':
            self._draw_box_wireframe(bb, draw_args)
        elif box_type == 'shaded_wireframe':
            self._draw_box_shaded(bb, draw_args['shaded'])
            self._draw_box_wireframe(bb, draw_args['wireframe'])
        else:
            raise ValueError(f'Unknown box type in draw_args: {box_type}; Must be one of [shaded, wireframe].')

    def _draw_box_shaded(self, bb: BoundingBox, draw_args: Dict) -> None:
        """
        Draws a shaded box to the current plot. Assumes draw_args contains legal data.

        :param bb: bounding box representing the box to draw
        :param draw_args: drawing args; passed to ax.plot_surface(...)
        :return: None
        """

        # https://stackoverflow.com/questions/33540109/plot-surfaces-on-a-cube

        def _plot(x, y, z):
            self.ax.plot_surface(x, y, z, **draw_args)

        constant_side = np.zeros((2, 1))

        # bottom face
        x, y = np.meshgrid([bb.p1.x, bb.p2.x], [bb.p1.y, bb.p2.y])
        constant_side.fill(bb.p1.z)
        _plot(x, y, constant_side)
        # top face
        constant_side.fill(bb.p2.z)
        _plot(x, y, constant_side)
        # pos X face
        y, z = np.meshgrid([bb.p1.y, bb.p2.y], [bb.p1.z, bb.p2.z])
        constant_side.fill(bb.p1.x)
        _plot(constant_side, y, z)
        # neg X face
        constant_side.fill(bb.p2.x)
        _plot(constant_side, y, z)
        # pos Y face
        x, z = np.meshgrid([bb.p1.x, bb.p2.x], [bb.p1.z, bb.p2.z])
        constant_side.fill(bb.p1.y)
        _plot(x, constant_side, z)
        # neg Y face
        constant_side.fill(bb.p2.y)
        _plot(x, constant_side, z)

    def _draw_box_wireframe(self, bb: BoundingBox, draw_args: Dict):
        """
        Draws a wireframe box to the current axes. Assumes draw_args contains legal data.

        :param bb: bounding box representing the box to draw
        :param draw_args: drawing args; passed to ax.plot(...)
        :return: None
        """

        def _plot_line3(x, y, z):
            self.ax.plot(x, y, z, **draw_args)

        delta_x = [bb.p1.x, bb.p2.x]
        delta_y = [bb.p1.y, bb.p2.y]
        delta_z = [bb.p1.z, bb.p2.z]
        x_lower = [bb.p1.x] * 2
        x_upper = [bb.p2.x] * 2
        y_lower = [bb.p1.y] * 2
        y_upper = [bb.p2.y] * 2
        z_lower = [bb.p1.z] * 2
        z_upper = [bb.p2.z] * 2
        _plot_line3(delta_x, y_lower, z_lower)
        _plot_line3(x_lower, delta_y, z_lower)
        _plot_line3(delta_x, y_upper, z_lower)
        _plot_line3(x_upper, delta_y, z_lower)
        _plot_line3(delta_x, y_lower, z_upper)
        _plot_line3(x_lower, delta_y, z_upper)
        _plot_line3(delta_x, y_upper, z_upper)
        _plot_line3(x_upper, delta_y, z_upper)
        _plot_line3(x_lower, y_lower, delta_z)
        _plot_line3(x_upper, y_lower, delta_z)
        _plot_line3(x_lower, y_upper, delta_z)
        _plot_line3(x_upper, y_upper, delta_z)

    def _draw_grid(self) -> None:
        """
        Draws the grid to the current axes.

        :return: None
        """

        def _draw_line3(p1_: Point3, p2_: Point3) -> None:
            self.ax.plot([p1_.x, p2_.x], [p1_.y, p2_.y], [p1_.z, p2_.z], **DRAW_ARGS_GRID)

        # x-direction lines
        for node_z in range(self.grid.node_count_z):
            for node_x in range(self.grid.node_count_x):
                p1 = self.grid.grid_to_xyz(node_x, 0, node_z)
                p2 = self.grid.grid_to_xyz(node_x, self.grid.node_count_y - 1, node_z)
                _draw_line3(p1, p2)

        # y-direction lines
        for node_z in range(self.grid.node_count_z):
            for node_y in range(self.grid.node_count_y):
                p1 = self.grid.grid_to_xyz(0, node_y, node_z)
                p2 = self.grid.grid_to_xyz(self.grid.node_count_x - 1, node_y, node_z)
                _draw_line3(p1, p2)

        # z-direction lines
        for node_y in range(self.grid.node_count_y):
            for node_x in range(self.grid.node_count_x):
                p1 = self.grid.grid_to_xyz(node_x, node_y, 0)
                p2 = self.grid.grid_to_xyz(node_x, node_y, self.grid.node_count_z - 1)
                _draw_line3(p1, p2)

    def _draw_pipe_routes(self, draw_args: Dict) -> None:
        """
        Draws all pipe routes on existing figure.

        :param draw_args: drawing args
        :return: None
        """
        for result in self.solver_results:
            pipe_route = result.best_pipe_route
            x = []
            y = []
            z = []
            for elbow in pipe_route.elbows:
                p_xyz = self.grid.grid_to_xyz(*elbow)
                x.append(p_xyz.x)
                y.append(p_xyz.y)
                z.append(p_xyz.z)

            draw_args['color'] = SOLVER_COLORS[result.solver_name]
            self.ax.plot(x, y, z, **draw_args)
            start = self.grid.grid_to_xyz(*pipe_route.anchor_start_pos)
            end = self.grid.grid_to_xyz(*pipe_route.anchor_end_pos)

            # draw special terminus markers
            self.ax.scatter(*start, marker='*', c='green', s=100)
            self.ax.scatter(*end, marker='*', c='red', s=100)

    def _draw_legend(self) -> None:
        """
        Draws a customized legend for each defined solver type.

        :return: None
        """
        # https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
        custom_lines = [Line2D([0], [0], color=SOLVER_COLORS[key], lw=4) for key in SOLVER_COLORS]
        self.ax.legend(custom_lines, [solver_name for solver_name in SOLVER_COLORS])


def test_empty_plot():
    viz = Visualizer()
    viz.render(show=True)


def test_single_box():
    viz = Visualizer()
    volume = BoundingBox(Point3(0, 0, 0), Point3(10, 30, 60))
    viz.set_volume(volume)
    viz._build_plot()
    bb = BoundingBox(Point3(2, 10, 20), Point3(8, 20, 50))
    draw_args = {'face_color': 'green', 'alpha': 0.5, 'edge_color': 'black', 'fill': True}
    viz._draw_box_shaded(bb, draw_args)
    viz.render(show=True, hold_on=True)


def test_volume():
    volume = BoundingBox(Point3(0, 0, 0), Point3(10, 20, 30))
    viz = Visualizer()
    viz.set_volume(volume)
    viz.render(show=True)


def test_grid():
    volume = BoundingBox(Point3(0, 0, 0), Point3(10, 20, 30))
    grid = Grid(volume.get_x_size(), volume.get_y_size(), volume.get_z_size(), 5, )
    viz = Visualizer()
    viz.set_volume(volume)
    viz.set_grid(grid)
    viz.render(show=True)


def test_all():
    volume = BoundingBox(Point3(0, 0, 0), Point3(10, 20, 30))
    grid = Grid(volume.get_x_size(), volume.get_y_size(), volume.get_z_size(), 5, )
    viz = Visualizer()
    viz.set_volume(volume)
    viz.set_grid(grid)
    occupied = BoundingBox(Point3(5, 10, 20), Point3(8, 15, 25))
    free = BoundingBox(Point3(2, 14, 3), Point3(8, 20, 7))
    viz.add_occupied_space(occupied)
    viz.add_free_space(free)
    viz.render(show=True)


if __name__ == '__main__':
    test_all()
