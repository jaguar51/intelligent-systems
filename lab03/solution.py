from math import log
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
from tabulate import tabulate


def read_data(dirname: str = './dataset') -> defaultdict:
    dataset = defaultdict(list)

    def read_file(file: Path) -> list:
        with file.open() as f:
            res = [int(word) for line in f for word in line.replace('Subject:', '').split()]

        return res

    def is_spam_msg(file: Path) -> int:
        if 'spmsg' in file.name:
            return 0
        return 1

    for part in Path(dirname).iterdir():
        if not part.is_dir():
            continue

        for msg in part.iterdir():
            if not msg.is_file():
                continue

            dataset[part.name].append((read_file(msg), is_spam_msg(msg)))

    return dataset


def train_test_split(dataset: defaultdict, k=10):
    i = 0
    for test_part in dataset.keys():
        if i >= k:
            return
        i += 1

        x_test, y_test, x_train, y_train = [], [], [], []

        for train_part in dataset.keys():
            data = dataset[train_part]
            for doc, label in data:
                if test_part == train_part:
                    x_test.append(doc)
                    y_test.append(label)
                else:
                    x_train.append(doc)
                    y_train.append(label)

        yield x_test, y_test, x_train, y_train


class NaiveBayes:
    def __init__(self):
        self._classes = defaultdict()
        self._freq = defaultdict()

    def fit(self, x: list, y: list):
        classes, freq = defaultdict(lambda: 0), defaultdict(lambda: 0)

        for i in range(len(x)):
            cur_class = y[i]
            words = x[i]

            classes[cur_class] += 1

            for word in words:
                freq[cur_class, word] += 1

        for cur_class, word in freq:
            freq[cur_class, word] /= classes[cur_class]

        for cur_class in classes:
            classes[cur_class] /= len(x)

        self._classes, self._freq = classes, freq
        return self

    def predict(self, x: list):
        y = []
        for doc in x:
            val = min(
                self._classes.keys(),
                key=lambda cl: -log(self._classes[cl]) + sum(
                    -log(self._freq.get((cl, word), 10 ** (-7))) for word in doc)
            )
            y.append(val)
        return y


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


def k_fold_cross_validation(dataset: defaultdict):
    total_f_mera = FMeraCalculator([0, 1])

    for x_test, y_test, x_train, y_train in train_test_split(dataset):
        local_f_mera = FMeraCalculator([0, 1])
        predict = NaiveBayes().fit(x_train, y_train).predict(x_test)
        for i in range(len(predict)):
            total_f_mera.add_data(predict[i], y_test[i])
            local_f_mera.add_data(predict[i], y_test[i])

        data = [[local_f_mera.__precision__(), local_f_mera.__recall__(), local_f_mera.get_mera()]]
        print(tabulate(data, headers=["precision", "recall", "F"], floatfmt='.3f'))
        print()

    print('Total F measure result: {}'.format(total_f_mera.get_mera()))


if __name__ == '__main__':
    data = read_data()
    k_fold_cross_validation(data)
