from typing import List, Tuple

from kernels import *
from calculators import *
from point import *


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
    train, test = get_data_sets()

    pass
