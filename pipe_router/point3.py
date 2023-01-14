"""
Defines the Point3 class.
"""
import math
from copy import deepcopy


class Point3:
    """
    Defines a 3D point data class.
    """

    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.x = x
        self.y = y
        self.z = z

    def round_to_int(self) -> 'Point3':
        """
        Rounds each coordinate in this point to the nearest integer.

        :return: reference to modified self
        """
        self.x = int(round(self.x))
        self.y = int(round(self.y))
        self.z = int(round(self.z))
        return self

    def distance_to(self, other: 'Point3', method: str = 'euclidean') -> float:
        """
        Computes the distance from this point to another point.

        :param other: other point to measure distance to
        :param method: distance method; must be one of ['e', 'euclidean', 'm', 'manhattan']
        :return: distance value
        """
        supported_methods = ['e', 'euclidean', 'm', 'manhattan']
        if method not in supported_methods:
            raise ValueError(f'Unknown distance method; expected one of {supported_methods}, got {method}')

        if method in ['e', 'euclidean']:
            return math.sqrt((self.x - other.x) ** 2 +
                             (self.y - other.y) ** 2 +
                             (self.z - other.z) ** 2)
        elif method in ['m', 'manhattan']:
            return (abs(self.x - other.x) +
                    abs(self.y - other.y) +
                    abs(self.z - other.z))

    def abs_sum(self) -> float:
        """
        Computes the absolute value of the sum of the elements.

        :return: absolute sum
        """
        return abs(self.x) + abs(self.y) + abs(self.z)

    def normalize(self) -> 'Point3':
        """
        Interprets the current object as a vector and returns a
        normalized copy (length 1). The original object is not modified.

        :return: normalized obj
        """
        p = deepcopy(self)
        length = p.distance_to(Point3.zero())
        p.x /= length
        p.y /= length
        p.z /= length
        return p

    @staticmethod
    def zero() -> 'Point3':
        """
        Returns a zeroed Point3 object.

        :return: point obj
        """
        return Point3(0, 0, 0)

    def __add__(self, other):
        if isinstance(other, Point3):
            return Point3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        if isinstance(other, Point3):
            return Point3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self):
        return Point3(-self.x, -self.y, -self.z)

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Point3(self.x * other, self.y * other, self.z * other)

    def __lt__(self, other):
        if isinstance(other, Point3):
            sqr_dist_self = self.x ** 2 + self.y ** 2 + self.z ** 2
            sqr_dist_other = other.x ** 2 + other.y ** 2 + other.z ** 2
            return sqr_dist_self < sqr_dist_other

    def __eq__(self, other):
        if isinstance(other, Point3):
            if self.x == other.x and self.y == other.y and self.z == other.z:
                return True
        return False

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __str__(self):
        return f'({self.x}, {self.y}, {self.z})'

    def __repr__(self):
        return self.__str__()
