import abc
import math
from typing import List

from point import Point


class Transformer(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def transform(self, left: Point) -> List[float]:
        pass


class DefTransformer(Transformer):
    def transform(self, point: Point) -> List[float]:
        return [point.x, point.y]


class RadialTransformer(Transformer):
    def transform(self, left: Point) -> List[float]:
        return [math.sqrt(left.x ** 2 + left.y ** 2), math.acos(left.x / math.sqrt(left.x ** 2 + left.y ** 2))]


class ConusTransformer(Transformer):
    def transform(self, left: Point) -> List[float]:
        return [left.x, left.y, math.sqrt(left.x**2 + left.y**2)]


class DistanceCalculator(metaclass=abc.ABCMeta):
    def __init__(self, transformer: Transformer = DefTransformer()):
        self.transformer = transformer

    @abc.abstractclassmethod
    def calculate(self, left: Point, right: Point) -> float:
        pass


class ManhattanCalculator(DistanceCalculator):
    def calculate(self, left: Point, right: Point) -> float:
        res = 0.0
        l_transf = self.transformer.transform(left)
        r_transf = self.transformer.transform(right)
        for i in range(len(r_transf)):
            res += math.fabs(l_transf[i] - r_transf[i])

        return res


class EuclidCalculator(DistanceCalculator):
    def calculate(self, left: Point, right: Point) -> float:
        res = 0.0
        l_transf = self.transformer.transform(left)
        r_transf = self.transformer.transform(right)
        for i in range(len(r_transf)):
            res += (l_transf[i] - r_transf[i]) ** 2

        return math.sqrt(res)
