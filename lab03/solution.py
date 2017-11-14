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
        return 0 if 'spmsg' in file.name else 1

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
    def __init__(self, threshold=0.0):
        self.threshold = threshold
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
        epsilon = np.finfo(float).eps

        for doc in x:
            values = defaultdict(lambda: 0)
            for cl in self._classes.keys():
                val = -log(self._classes[cl])
                val += sum(-log(self._freq.get((cl, word), epsilon)) for word in doc)
                values[cl] += val

            # y.append(min(values, key=values.get))
            y.append(self._class_2(values))
        return y

    def _class_2(self, values):
        if values[0] - values[1] < self.threshold:
            return 0
        else:
            return 1


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


def k_fold_cross_validation(dataset: defaultdict, threshold, enable_log=False):
    total_f_mera = FMeraCalculator([0, 1])

    for x_test, y_test, x_train, y_train in train_test_split(dataset):
        local_f_mera = FMeraCalculator([0, 1])
        predict = NaiveBayes(threshold).fit(x_train, y_train).predict(x_test)
        for i in range(len(predict)):
            total_f_mera.add_data(predict[i], y_test[i])
            local_f_mera.add_data(predict[i], y_test[i])

        if enable_log:
            data = [[
                local_f_mera.matrix[1, 1],
                local_f_mera.matrix[0, 0],
                local_f_mera.matrix[0, 1],
                local_f_mera.matrix[1, 0],
                local_f_mera.get_mera()
            ]]
            print(tabulate(data, headers=["TP", "TN", "FP", "FN", "F"], floatfmt='.3f'))
            print()

    print("-------------------------------------------")
    print("Total")
    data = [[
        total_f_mera.matrix[1, 1],
        total_f_mera.matrix[0, 0],
        total_f_mera.matrix[0, 1],
        total_f_mera.matrix[1, 0],
        total_f_mera.get_mera()
    ]]
    print(tabulate(data, headers=["TP", "TN", "FP", "FN", "F"], floatfmt='.3f'))
    print('Total F measure result: {}'.format(total_f_mera.get_mera()))


def find_threshold(dataset: defaultdict, max=400):
    left = 0
    right = max

    for i in range(100):
        mid = (left + right) / 2

        total_f_mera = FMeraCalculator([0, 1])
        for x_test, y_test, x_train, y_train in train_test_split(dataset):
            local_f_mera = FMeraCalculator([0, 1])
            predict = NaiveBayes(mid).fit(x_train, y_train).predict(x_test)
            for i in range(len(predict)):
                total_f_mera.add_data(predict[i], y_test[i])
                local_f_mera.add_data(predict[i], y_test[i])

        # mera = total_f_mera.get_mera()
        fn = total_f_mera.matrix[1, 0]
        print(mid)
        if fn > 0:
            left = mid
        else:
            right = mid

    print(left, right)


if __name__ == '__main__':
    data = read_data()
    k_fold_cross_validation(data, 0)
    k_fold_cross_validation(data, 305.78954815864563)
    k_fold_cross_validation(data, 310)
    # find_threshold(data)
