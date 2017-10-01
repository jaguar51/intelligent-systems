import abc
import math
from point import Point


class DistanceCalculator(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def calculate(self, left: Point, right: Point) -> float:
        pass


class ManhattanCalculator(DistanceCalculator):
    def calculate(self, left: Point, right: Point) -> float:
        return math.fabs((left.x - right.x) + (left.y - right.y))


class EuclidCalculator(DistanceCalculator):
    def calculate(self, left: Point, right: Point) -> float:
        return math.sqrt((left.x - right.x) ** 2 + (left.y - right.y) ** 2)
