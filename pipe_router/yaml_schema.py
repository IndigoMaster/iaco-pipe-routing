"""
The file defines the YAML configuration schema.
"""
from schema import Schema, And, Or, Optional

from solvers.acs import SolverACS
from solvers.bacs import SolverBACS
from solvers.iaco import SolverIACO
from solvers.solver import Solver
from solvers.solver_base import SolverBase


# region CONSTANTS
class YAMLKeys:
    SCENARIO_TITLE: str = 'title'

    class Grid:
        KEY_NAME: str = 'layout_volume'
        VOLUME_SIZE_X: str = 'true_size_x'
        VOLUME_SIZE_Y: str = 'true_size_y'
        VOLUME_SIZE_Z: str = 'true_size_z'
        UNIT_GRID_SIZE: str = 'grid_size'

    class BoundingBoxes:
        BOX_ID: str = 'name'

        class OccupiedSpaces:
            KEY_NAME = 'occupied_space_objects'
            PT1_X = 'corner1_x'
            PT1_Y = 'corner1_y'
            PT1_Z = 'corner1_z'
            PT2_X = 'corner2_x'
            PT2_Y = 'corner2_y'
            PT2_Z = 'corner2_z'

        class FreeSpaces:
            KEY_NAME = 'free_space_objects'
            PT1_X = 'corner1_x'
            PT1_Y = 'corner1_y'
            PT1_Z = 'corner1_z'
            PT2_X = 'corner2_x'
            PT2_Y = 'corner2_y'
            PT2_Z = 'corner2_z'

    class PipeRoute:
        KEY_NAME = 'pipe_connections'
        ROUTE_ID = 'name'
        PT1_X = 'start_x'
        PT1_Y = 'start_y'
        PT1_Z = 'start_z'
        PT2_X = 'end_x'
        PT2_Y = 'end_y'
        PT2_Z = 'end_z'

    class Solver:
        KEY_NAME = 'solver'


# endregion

def positive_numerical(*, int_only: bool = False) -> And:
    """
    Creates a Schema And object to restrict values to
    numerical types (float and/or int) >= 0.

    :param int_only: if True, restricts to int, else int or float
    :return: And object
    """
    _type = int if int_only else Or(float, int)
    return And(_type, lambda x: x >= 0)


solver_base_schema = {
    Solver.ArgMap.ANT_COUNT: positive_numerical(int_only=True),
    Solver.ArgMap.ALPHA: positive_numerical(),
    Solver.ArgMap.BETA: positive_numerical(),
    Solver.ArgMap.WEIGHT_ROUTE_LEN: positive_numerical(),
    Solver.ArgMap.WEIGHT_ELBOW_COUNT: positive_numerical(),
    Solver.ArgMap.WEIGHT_ROUTE_EVAL: positive_numerical(),
    Solver.ArgMap.ITER_COUNT: positive_numerical(int_only=True)
}

config_schema = Schema({
    YAMLKeys.SCENARIO_TITLE: str,
    YAMLKeys.Grid.KEY_NAME: {
        YAMLKeys.Grid.VOLUME_SIZE_X: positive_numerical(),
        YAMLKeys.Grid.VOLUME_SIZE_Y: positive_numerical(),
        YAMLKeys.Grid.VOLUME_SIZE_Z: positive_numerical(),
        YAMLKeys.Grid.UNIT_GRID_SIZE: positive_numerical(),
    },
    Optional(YAMLKeys.BoundingBoxes.OccupiedSpaces.KEY_NAME, default=[]): [{
        YAMLKeys.BoundingBoxes.BOX_ID: str,
        YAMLKeys.BoundingBoxes.OccupiedSpaces.PT1_X: positive_numerical(),
        YAMLKeys.BoundingBoxes.OccupiedSpaces.PT1_Y: positive_numerical(),
        YAMLKeys.BoundingBoxes.OccupiedSpaces.PT1_Z: positive_numerical(),
        YAMLKeys.BoundingBoxes.OccupiedSpaces.PT2_X: positive_numerical(),
        YAMLKeys.BoundingBoxes.OccupiedSpaces.PT2_Y: positive_numerical(),
        YAMLKeys.BoundingBoxes.OccupiedSpaces.PT2_Z: positive_numerical()
    }],
    Optional(YAMLKeys.BoundingBoxes.FreeSpaces.KEY_NAME, default=[]): [{
        YAMLKeys.BoundingBoxes.BOX_ID: str,
        YAMLKeys.BoundingBoxes.FreeSpaces.PT1_X: positive_numerical(),
        YAMLKeys.BoundingBoxes.FreeSpaces.PT1_Y: positive_numerical(),
        YAMLKeys.BoundingBoxes.FreeSpaces.PT1_Z: positive_numerical(),
        YAMLKeys.BoundingBoxes.FreeSpaces.PT2_X: positive_numerical(),
        YAMLKeys.BoundingBoxes.FreeSpaces.PT2_Y: positive_numerical(),
        YAMLKeys.BoundingBoxes.FreeSpaces.PT2_Z: positive_numerical()
    }],
    Optional(YAMLKeys.PipeRoute.KEY_NAME, default=[]): [{
        YAMLKeys.PipeRoute.ROUTE_ID: str,
        YAMLKeys.PipeRoute.PT1_X: positive_numerical(),
        YAMLKeys.PipeRoute.PT1_Y: positive_numerical(),
        YAMLKeys.PipeRoute.PT1_Z: positive_numerical(),
        YAMLKeys.PipeRoute.PT2_X: positive_numerical(),
        YAMLKeys.PipeRoute.PT2_Y: positive_numerical(),
        YAMLKeys.PipeRoute.PT2_Z: positive_numerical(),
    }],
    Optional(YAMLKeys.Solver.KEY_NAME, default=[]): [Or(
        solver_base_schema | {
            SolverBase.ArgMap.SOLVER_NAME_VAR: SolverACS.ArgMap.SOLVER_NAME,
            Solver.ArgMap.Q0: positive_numerical(),
            Solver.ArgMap.RHO: positive_numerical()
        },
        solver_base_schema | {
            SolverBase.ArgMap.SOLVER_NAME_VAR: SolverBACS.ArgMap.SOLVER_NAME,
            SolverBACS.ArgMap.Q_MIN: positive_numerical(),
            SolverBACS.ArgMap.Q_MAX: positive_numerical(),
            SolverBACS.ArgMap.RHO_MIN: positive_numerical(),
            SolverBACS.ArgMap.RHO_MAX: positive_numerical()
        },
        solver_base_schema | {
            SolverBase.ArgMap.SOLVER_NAME_VAR: SolverIACO.ArgMap.SOLVER_NAME,
            SolverIACO.ArgMap.Q_MIN: positive_numerical(),
            SolverIACO.ArgMap.Q_MAX: positive_numerical(),
            SolverIACO.ArgMap.RHO_MIN: positive_numerical(),
            SolverIACO.ArgMap.RHO_MAX: positive_numerical()
        })]
}, ignore_extra_keys=True)
