from typing import List, Tuple

import operator
from kernels import *
from calculators import *
from point import *


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
                self.dist_calc.calculate(neighbour, test) / self.dist_calc.calculate(neighbour, neighbour_k_1)
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


def read_test_file() -> List[Point]:
    res = []
    with open('dataset.txt') as f:
        for line in f:
            values = line.split(',')
            res.append(Point(float(values[0]), float(values[1]), int(values[2])))

    return res


def get_data_sets(fold_count: int = 5) -> Tuple[List[Point], List[Point]]:
    train = []
    test = []

    c: int = 0
    for point in read_test_file():
        if c == fold_count - 1:
            test.append(point)
            c = 0
        else:
            train.append(point)
            c += 1

    return train, test


if __name__ == '__main__':
    # train, test = get_data_sets(fold_count=5)
    # neighbour = KNearesNeighbour(train=train, k=5, kernel=GausKernel()) -> best!!!

    # train, test = get_data_sets(fold_count=5)
    # neighbour = KNearesNeighbour(train=train, k=6, kernel=GausKernel())

    train, test = get_data_sets(fold_count=5)
    neighbour = KNearesNeighbour(train=train, k=6, kernel=GausKernel())
    for t in test:
        predict_cat = neighbour.predict(t)
        true_cat = t.category
        print(predict_cat, true_cat)

    pass
