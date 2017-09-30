import abc
import math


class Kernel(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def calculate(self, r: float) -> float:
        pass


class GausKernel(Kernel):
    def calculate(self, r: float) -> float:
        return (2 * math.pi) ** (-0.5) * math.e ** (-0.5 * r ** 2)


class TriangleKernel(Kernel):
    def calculate(self, r: float) -> float:
        return 1 - abs(r)
