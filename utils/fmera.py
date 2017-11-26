import numpy as np


class FMeraCalculator:
    def __init__(self, categories):
        self._categories = dict()

        ind = 0
        for cat in categories:
            self._categories[cat] = ind
            ind += 1

        self._categories_count = len(categories)
        self._matrix = np.zeros((self._categories_count, self._categories_count))

    def add_data(self, actual, expected):
        self._matrix[self._get_index(actual)][self._get_index(expected)] += 1

    def get_mera(self):
        precision = self._precision()
        recall = self._recall()
        return 2 * (precision * recall) / (precision + recall)

    def _get_index(self, cat: int):
        return self._categories[cat]

    def _recall(self):
        recall = 0.0
        for i in range(self._categories_count):
            sum_by_column = 0.0
            for j in range(self._categories_count):
                sum_by_column += self._matrix[j][i]

            recall += (self._matrix[i][i] / sum_by_column)

        return recall / self._categories_count

    def _precision(self):
        precision = 0.0
        for i in range(self._categories_count):
            sum_by_column = 0.0
            for j in range(self._categories_count):
                sum_by_column += self._matrix[i][j]

            if sum_by_column == 0:
                precision += 0
            else:
                precision += (self._matrix[i][i] / sum_by_column)

        return precision / self._categories_count
