"""
BoundingBox class definition.
"""

from pipe_router.point3 import Point3


class BoundingBox:
    """
    Represents an axis-oriented (upright) 3D bounding box defined by two 3D points.
    """

    def __init__(self, p1: Point3, p2: Point3, *, name: str = ''):
        """
        Defines a bounding box from two 3D points. Point order does not matter.

        :param p1: first corner point
        :param p2: second corner point
        :param name: optional; string name for this bounding box
        """
        self.name = name
        self.p1 = p1
        self.p2 = p2

        # ensure p1 is "smallest" (closer to origin)
        if p2 < p1:
            self.p1, self.p2 = self.p2, self.p1

    def contains(self, p: Point3, surface_inclusive: bool = True) -> bool:
        """
        Tests if this bounding box contains the specified point.

        :param p: point to test if inside bounding box
        :param surface_inclusive: if True, a point located exactly on the
            box surface is considered to be "contained" in the box.
        :return: True if point is located inside the bounding box, else False
        """
        if surface_inclusive:
            return self.p1.x <= p.x <= self.p2.x and \
                self.p1.y <= p.y <= self.p2.y and \
                self.p1.z <= p.z <= self.p2.z
        else:
            return self.p1.x < p.x < self.p2.x and \
                self.p1.y < p.y < self.p2.y and \
                self.p1.z < p.z < self.p2.z

    def is_on_surface(self, p: Point3) -> bool:
        """
        Tests if the given point lies exactly on any surface of this bounding box.

        :param p: point to test
        :return: True if point lies on bounding box surface
        """
        return self.contains(p, surface_inclusive=True) and not self.contains(p, surface_inclusive=False)

    def get_x_size(self) -> float:
        """
        Gets the size of the bounding box in the x-direction.

        :return: box size in X
        """
        return self.p2.x - self.p1.x

    def get_y_size(self) -> float:
        """
        Gets the size of the bounding box in the y-direction.

        :return: box size in Y
        """
        return self.p2.y - self.p1.y

    def get_z_size(self) -> float:
        """
        Gets the size of the bounding box in the z-direction.

        :return: box size in Z
        """
        return self.p2.z - self.p1.z

    def get_normal_direction(self, point_on_surface: Point3) -> Point3:
        """
        Determines the normal direction given a point located on the bounding box.

        :param point_on_surface: point on surface
        :return: point representing normal direction
        """
        directions = [
            Point3(-1, 0, 0),
            Point3(1, 0, 0),
            Point3(0, -1, 0),
            Point3(0, 1, 0),
            Point3(0, 0, -1),
            Point3(0, 0, 1),
        ]

        for direction in directions:
            candidate = point_on_surface + (direction * 0.01)
            if self.contains(candidate, surface_inclusive=False):
                return -direction

        raise ValueError(f'Point {point_on_surface} is not located on the surface of the bounding box {self}')

    def passes_through(self, p1: Point3, p2: Point3) -> bool:
        """
        Determines if the straight line connecting two points
        passes through this bounding box. Assumes both points
        lie outside or on surface of bounding box and connecting
        line is orthogonal to 3D axes.

        :param p1: first point
        :param p2: second point
        :return: True if connecting line passes through bounding box
        """
        # pass through in X direction
        if (p1.y <= self.p1.y) and (self.p2.y <= p2.y) and (p1.z <= self.p1.z) and (self.p2.z <= p2.z):
            return True
        # pass through in Y direction
        if (p1.x <= self.p1.x) and (self.p2.x <= p2.x) and (p1.z <= self.p1.z) and (self.p2.z <= p2.z):
            return True
        # pass through in Z direction
        if (p1.x <= self.p1.x) and (self.p2.x <= p2.x) and (p1.y <= self.p1.y) and (self.p2.y <= p2.y):
            return True
        return False

    def __str__(self):
        return f'BB(p1={self.p1}, p2={self.p2})'

    def __repr__(self):
        return self.__str__()
