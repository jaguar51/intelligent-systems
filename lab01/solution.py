from typing import List, Tuple

import operator
from lab01.kernels import *
from lab01.calculators import *
from lab01.point import *
import numpy as np

import matplotlib.pyplot as plt


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

            if sum_by_column == 0:
                precision += 0
            else:
                precision += (self.matrix[i][i] / sum_by_column)

        return precision / self.categories_count


def read_test_file(shuffle: bool = False):
    res = []
    with open('dataset.txt') as f:
        for line in f:
            values = line.split(',')
            res.append(Point(float(values[0]), float(values[1]), int(values[2])))

    if shuffle:
        np.random.shuffle(res)
    return res


def get_data_sets(train, test, fold_count: int = 5) -> Tuple[List[Point], List[Point]]:
    all_data = []
    all_data.extend(train)
    all_data.extend(test)
    return all_data[len(all_data) // fold_count:], all_data[0: len(all_data) // fold_count]


def test(
        kernel: Kernel,
        dist_calc: DistanceCalculator,
        dataset,
        k: int,
        fold_count: int = 5
):
    f_mera = FMeraCalculator([0, 1])

    train = dataset
    test = []
    for i in range(len(dataset) // fold_count):
        train, test = get_data_sets(train, test, fold_count)
        neighbour = KNearesNeighbour(train=train, k=k, kernel=GausKernel(), dist_calc=dist_calc)
        for t in test:
            predict_cat = neighbour.predict(t)
            true_cat = t.category
            f_mera.add_data(predict_cat, true_cat)

    return f_mera.get_mera()


if __name__ == '__main__':
    dataset = read_test_file(True)

    # for fold_count in range(2, 20):
    x = []
    y = []
    for k in range(2, 20):
        f = test(GausKernel(), EuclidCalculator(ConusTransformer()), dataset, k, 10)
        x.append(k)
        y.append(f)
        print('k={}; f={}'.format(k, f))

    plt.plot(x, y)
    plt.show()

    pass
