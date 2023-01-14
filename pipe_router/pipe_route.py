"""
PipeRoute class definition.
"""

from typing import List, Union

from pipe_router.point3 import Point3


class PipeRoute:
    """
    Represents a single pipe route, with start and end positions
    and other characteristics.
    """

    def __init__(self, *, start_point: Point3, end_point: Point3, name: str = ''):
        self.name = name
        self.routing_start_pos = start_point
        self.routing_end_pos = end_point
        self.elbows: List[Point3] = [start_point, end_point]
        self.start_normal_dir: Union[Point3, None] = None
        self.end_normal_dir: Union[Point3, None] = None
        self.anchor_start_pos: Point3 = start_point
        self.anchor_end_pos: Point3 = end_point

    def update_terminals_from_normals(self, start_pos_normal_dir: Point3, end_pos_normal_dir: Point3, unit_grid_size: float) -> None:
        """
        Computes the new start and end positions of this pipe route using the
        normal direction information relative to bounding boxes.

        :param start_pos_normal_dir: normal dir for start position; can be None
        :param end_pos_normal_dir: normal dir for end position; can be None
        :param unit_grid_size: unit grid size
        :return: None
        """
        start_pos_normal_dir = start_pos_normal_dir.normalize().round_to_int()
        end_pos_normal_dir = end_pos_normal_dir.normalize().round_to_int()
        self.start_normal_dir = start_pos_normal_dir
        self.end_normal_dir = end_pos_normal_dir
        self.anchor_start_pos = self.routing_start_pos
        self.anchor_end_pos = self.routing_end_pos
        if start_pos_normal_dir is not None:
            self.routing_start_pos += start_pos_normal_dir * unit_grid_size
        if end_pos_normal_dir is not None:
            self.routing_end_pos += end_pos_normal_dir * unit_grid_size

    def __str__(self):
        return f'{__class__.__name__}({self.anchor_start_pos} to {self.anchor_end_pos})'

    def __repr__(self):
        return self.__str__()
