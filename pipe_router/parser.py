"""
This file defines operations for reading, writing, and parsing YAML to convert to training parameters.
"""

import logging
from pathlib import Path
from typing import Dict, List

import yaml

from pipe_router.bounding_box import BoundingBox
from pipe_router.grid import Grid
from pipe_router.params import Params
from pipe_router.pipe_route import PipeRoute
from pipe_router.point3 import Point3
from pipe_router.solvers.acs import SolverACS
from pipe_router.solvers.bacs import SolverBACS
from pipe_router.solvers.iaco import SolverIACO
from pipe_router.solvers.solver_base import SolverBase
from pipe_router.yaml_schema import config_schema, YAMLKeys
from solvers.solver import Solver


def get_attr_list(obj) -> List[str]:
    """
    Gets the public attribute list of an object.

    :param obj: object instance to inspect
    :return: list of strings
    """
    return [attr for attr in dir(obj) if not attr.startswith('_')]


def parse_yaml(yaml_path: Path) -> Params:
    """
    Parses a YAML config file, performs basic validation, and
    generates a Params object containing the config data.

    :param yaml_path: Path to YAML file
    :return: Params object
    """
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)
        if yaml_dict is not None:
            try:
                data = config_schema.validate(yaml_dict)
                logging.info('Configuration file is valid')
                params = _yaml_to_params(data)
                return params
            except Exception as e:
                logging.error('A failure occurred during YAML parsing. The following error may help you troubleshoot your YAML file.')
                raise e
        else:
            raise ValueError(f'Failed to read YAML file at {yaml_path}')


def _yaml_to_params(data: Dict) -> Params:
    """
    Builds and populates a Params object from the validated YAML data definitions.

    :param data: validated YAML data dictionary
    :return: constructed Params object
    """
    params = Params()
    params.set_scenario_title(data[YAMLKeys.SCENARIO_TITLE])
    grid = _build_layout_volume(data)
    params.set_grid(grid)
    params.set_volume(grid.true_3d_size)

    occupied_bb_list = _build_occupied_space_boxes(data)
    for bb in occupied_bb_list:
        params.add_occupied_space_box(bb)

    free_bb_list = _build_free_space_boxes(data)
    for bb in free_bb_list:
        params.add_free_space_box(bb)

    connection_list = _build_connections(data)
    for conn in connection_list:
        params.add_pipe_route(conn)

    solver_list = _build_solvers(data, params.grid)
    for solver in solver_list:
        params.add_solver(solver)

    return params


def _build_layout_volume(data: Dict) -> Grid:
    """
    Builds and configures a volume object.

    :param data: validated data dict
    :return: Grid obj
    """
    layout_def = data[YAMLKeys.Grid.KEY_NAME]
    x = layout_def[YAMLKeys.Grid.VOLUME_SIZE_X]
    y = layout_def[YAMLKeys.Grid.VOLUME_SIZE_Y]
    z = layout_def[YAMLKeys.Grid.VOLUME_SIZE_Z]
    unit_grid_size = layout_def[YAMLKeys.Grid.UNIT_GRID_SIZE]
    grid_true_size = Point3(x, y, z)
    grid = Grid(grid_true_size, unit_grid_size)
    return grid


def _build_occupied_space_boxes(data: Dict) -> List[BoundingBox]:
    """
    Constructs occupied space bounding box objects from YAML data.

    :param data: validated YAML data dictionary
    :return: list of bounding boxes
    """
    bb_list = []
    for bb_yaml in data[YAMLKeys.BoundingBoxes.OccupiedSpaces.KEY_NAME]:
        name = bb_yaml[YAMLKeys.BoundingBoxes.BOX_ID]
        x1 = bb_yaml[YAMLKeys.BoundingBoxes.OccupiedSpaces.PT1_X]
        y1 = bb_yaml[YAMLKeys.BoundingBoxes.OccupiedSpaces.PT1_Y]
        z1 = bb_yaml[YAMLKeys.BoundingBoxes.OccupiedSpaces.PT1_Z]
        x2 = bb_yaml[YAMLKeys.BoundingBoxes.OccupiedSpaces.PT2_X]
        y2 = bb_yaml[YAMLKeys.BoundingBoxes.OccupiedSpaces.PT2_Y]
        z2 = bb_yaml[YAMLKeys.BoundingBoxes.OccupiedSpaces.PT2_Z]
        p1 = Point3(x1, y1, z1)
        p2 = Point3(x2, y2, z2)
        bb = BoundingBox(p1, p2, name=name)
        bb_list.append(bb)
    return bb_list


def _build_free_space_boxes(data: Dict) -> List[BoundingBox]:
    """
    Constructs free space bounding box objects from YAML data.

    :param data: validated YAML data dictionary
    :return: list of bounding boxes
    """
    bb_list = []
    for bb_yaml in data[YAMLKeys.BoundingBoxes.FreeSpaces.KEY_NAME]:
        name = bb_yaml[YAMLKeys.BoundingBoxes.BOX_ID]
        x1 = bb_yaml[YAMLKeys.BoundingBoxes.FreeSpaces.PT1_X]
        y1 = bb_yaml[YAMLKeys.BoundingBoxes.FreeSpaces.PT1_Y]
        z1 = bb_yaml[YAMLKeys.BoundingBoxes.FreeSpaces.PT1_Z]
        x2 = bb_yaml[YAMLKeys.BoundingBoxes.FreeSpaces.PT2_X]
        y2 = bb_yaml[YAMLKeys.BoundingBoxes.FreeSpaces.PT2_Y]
        z2 = bb_yaml[YAMLKeys.BoundingBoxes.FreeSpaces.PT2_Z]
        p1 = Point3(x1, y1, z1)
        p2 = Point3(x2, y2, z2)
        bb = BoundingBox(p1, p2, name=name)
        bb_list.append(bb)
    return bb_list


def _build_connections(data: Dict) -> List[PipeRoute]:
    """
    Constructs connection objects from YAML data.

    :param data: validated YAML data dictionary
    :return: list of connection objects
    """
    route_list = []
    for yaml_route in data[YAMLKeys.PipeRoute.KEY_NAME]:
        name = yaml_route[YAMLKeys.PipeRoute.ROUTE_ID]
        start_x = yaml_route[YAMLKeys.PipeRoute.PT1_X]
        start_y = yaml_route[YAMLKeys.PipeRoute.PT1_Y]
        start_z = yaml_route[YAMLKeys.PipeRoute.PT1_Z]
        end_x = yaml_route[YAMLKeys.PipeRoute.PT2_X]
        end_y = yaml_route[YAMLKeys.PipeRoute.PT2_Y]
        end_z = yaml_route[YAMLKeys.PipeRoute.PT2_Z]
        start = Point3(start_x, start_y, start_z)
        end = Point3(end_x, end_y, end_z)
        c = PipeRoute(start_point=start, end_point=end, name=name)
        route_list.append(c)
    return route_list


def _build_solvers(data, grid: Grid) -> List[SolverBase]:
    """
    Extracts and organizes solver definitions from YAML data.
    (Does not construct solver objects.)

    :param data: validated YAML data
    :param grid: grid obj
    :return: list of solver definitions; elements are Dicts as {param_name, value}
    """
    solvers = []
    for solver_yaml in data[YAMLKeys.Solver.KEY_NAME]:
        # main solver args
        args = {
            Solver.ArgMap.ANT_COUNT: solver_yaml[Solver.ArgMap.ANT_COUNT],
            Solver.ArgMap.ITER_COUNT: solver_yaml[Solver.ArgMap.ITER_COUNT],
            Solver.ArgMap.ALPHA: solver_yaml[Solver.ArgMap.ALPHA],
            Solver.ArgMap.BETA: solver_yaml[Solver.ArgMap.BETA],
            Solver.ArgMap.WEIGHT_ROUTE_LEN: solver_yaml[Solver.ArgMap.WEIGHT_ROUTE_LEN],
            Solver.ArgMap.WEIGHT_ELBOW_COUNT: solver_yaml[Solver.ArgMap.WEIGHT_ELBOW_COUNT],
            Solver.ArgMap.WEIGHT_ROUTE_EVAL: solver_yaml[Solver.ArgMap.WEIGHT_ROUTE_EVAL],
            Solver.ArgMap.GRID: grid
        }
        # get solver-specific args
        if solver_yaml[SolverBase.ArgMap.SOLVER_NAME_VAR] == SolverACS.ArgMap.SOLVER_NAME:
            args[SolverACS.ArgMap.Q0]: solver_yaml[SolverACS.ArgMap.Q0]
            args[SolverACS.ArgMap.RHO]: solver_yaml[SolverACS.ArgMap.RHO]
            solver = SolverACS(**args)
        elif solver_yaml[SolverBase.ArgMap.SOLVER_NAME_VAR] == SolverBACS.ArgMap.SOLVER_NAME:
            args[SolverBACS.ArgMap.Q_MIN] = solver_yaml[SolverBACS.ArgMap.Q_MIN]
            args[SolverBACS.ArgMap.Q_MAX] = solver_yaml[SolverBACS.ArgMap.Q_MAX]
            args[SolverBACS.ArgMap.RHO_MIN] = solver_yaml[SolverBACS.ArgMap.RHO_MIN]
            args[SolverBACS.ArgMap.RHO_MAX] = solver_yaml[SolverBACS.ArgMap.RHO_MAX]
            solver = SolverBACS(**args)
        elif solver_yaml[SolverBase.ArgMap.SOLVER_NAME_VAR] == SolverIACO.ArgMap.SOLVER_NAME:
            args[SolverIACO.ArgMap.Q_MIN] = solver_yaml[SolverIACO.ArgMap.Q_MIN]
            args[SolverIACO.ArgMap.Q_MAX] = solver_yaml[SolverIACO.ArgMap.Q_MAX]
            args[SolverIACO.ArgMap.RHO_MIN] = solver_yaml[SolverIACO.ArgMap.RHO_MIN]
            args[SolverIACO.ArgMap.RHO_MAX] = solver_yaml[SolverIACO.ArgMap.RHO_MAX]
            solver = SolverIACO(**args)
        else:
            raise ValueError('Insufficient validation: bad solver name reached solver building logic')

        solvers.append(solver)
    return solvers
