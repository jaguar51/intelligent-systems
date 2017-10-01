from typing import List, Tuple

import operator
from kernels import *
from calculators import *
from point import *
import numpy as np


class KNearesNeighbour:
    def __init__(
            self,
            k: int = 5,
            train: List[Point] = [],
            kernel: Kernel = TriangleKernel(),
            dist_calc: DistanceCalculator = EuclidCalculator()
    ):
        self.k = k
        self.train_data = train
        self.kernel = kernel
        self.dist_calc = dist_calc

    def predict(self, test: Point) -> int:
        neighbours, neighbour_k_1 = self._find_nearest(test)

        cat_weight_map = {}
        for neighbour in neighbours:
            weight = self.kernel.calculate(
                self.dist_calc.calculate(test, neighbour) / self.dist_calc.calculate(test, neighbour_k_1)
            )
            category = neighbour.category

            if cat_weight_map.get(category) is None:
                cat_weight_map[category] = weight
            else:
                cat_weight_map[category] += weight

        return sorted(cat_weight_map.items(), key=operator.itemgetter(1), reverse=True)[0][0]

    def _find_nearest(self, test: Point):
        distances = []
        for train in self.train_data:
            dist = self.dist_calc.calculate(test, train)
            distances.append((train, dist))

        distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for x in range(self.k):
            neighbors.append(distances[x][0])

        return neighbors, distances[self.k][0]


class FMeraCalculator:
    def __init__(self, categories: List[int]):
        self.categories = categories
        self.categories_count = len(categories)
        self.matrix = np.zeros((self.categories_count, self.categories_count))

    def add_data(self, actual: int, expected: int):
        self.matrix[actual][expected] += 1

    def get_mera(self):
        precision = self.__precision__()
        recall = self.__recall__()
        return 2 * (precision * recall) / (precision + recall)

    def __recall__(self):
        recall = 0.0
        for i in range(self.categories_count):
            sum_by_column = 0.0
            for j in range(self.categories_count):
                sum_by_column += self.matrix[j][i]

            recall += (self.matrix[i][i] / sum_by_column)

        return recall / self.categories_count

    def __precision__(self):
        precision = 0.0
        for i in range(self.categories_count):
            sum_by_column = 0.0
            for j in range(self.categories_count):
                sum_by_column += self.matrix[i][j]

            precision += (self.matrix[i][i] / sum_by_column)

        return precision / self.categories_count


def read_test_file() -> List[Point]:
    res = []
    with open('dataset.txt') as f:
        for line in f:
            values = line.split(',')
            res.append(Point(float(values[0]), float(values[1]), int(values[2])))

    return res


def get_data_sets(fold_count: int = 5, shuffle: bool = False) -> Tuple[List[Point], List[Point]]:
    train = []
    test = []

    points = read_test_file()

    if shuffle:
        np.random.shuffle(points)

    c: int = 0
    for point in points:
        if c == fold_count - 1:
            test.append(point)
            c = 0
        else:
            train.append(point)
            c += 1

    return train, test


if __name__ == '__main__':
    # train, test = get_data_sets(fold_count=8)
    # neighbour = KNearesNeighbour(train=train, k=6, kernel=GausKernel()) # -> best 0.872

    # train, test = get_data_sets(fold_count=5)
    # neighbour = KNearesNeighbour(train=train, k=5, kernel=GausKernel()) # -> best 0.826

    # train, test = get_data_sets(fold_count=5)
    # neighbour = KNearesNeighbour(train=train, k=6, kernel=GausKernel()) # -> 0.78

    train, test = get_data_sets(fold_count=10, shuffle=True)
    neighbour = KNearesNeighbour(train=train, k=6, kernel=GausKernel())

    calculator = FMeraCalculator([0, 1])
    for t in test:
        predict_cat = neighbour.predict(t)
        true_cat = t.category
        calculator.add_data(predict_cat, true_cat)
        print(predict_cat, true_cat)

    print(calculator.get_mera())
    pass
