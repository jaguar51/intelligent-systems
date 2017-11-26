import abc

import cvxopt
import math

import numpy as np
import numpy.linalg as la

from lab03.solution import FMeraCalculator


class Transformer(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def transform(self, x, y):
        pass


class DefTransformer(Transformer):
    def transform(self, x, y):
        return x, y


class ConusTransformer(Transformer):
    def transform(self, x, y):
        n_x = [x[0], x[1], math.sqrt(x[0] ** 2 + x[1] ** 2)]
        n_y = [y[0], y[1], math.sqrt(y[0] ** 2 + y[1] ** 2)]

        return n_x, n_y


class Kernel(metaclass=abc.ABCMeta):
    def __init__(self, transformer: Transformer = DefTransformer()):
        self._transformer = transformer

    def calculate(self, x, y) -> float:
        n_x, n_y = self._transformer.transform(x, y)
        return self._calculate(n_x, n_y)

    @abc.abstractclassmethod
    def _calculate(self, x, y) -> float:
        pass


class LinearKernel(Kernel):
    def _calculate(self, x, y) -> float:
        return np.inner(x, y)


class RadialBasisKernel(Kernel):
    def __init__(self, transformer: Transformer = DefTransformer(), gamma=10) -> None:
        super().__init__(transformer)
        self._gamma = gamma

    def _calculate(self, x, y) -> float:
        return np.exp(-self._gamma * la.norm(np.subtract(x, y)))


class SVMTrainer:
    def __init__(self, kernel: Kernel = RadialBasisKernel(transformer=ConusTransformer()), c=0.1):
        self._kernel = kernel
        self._c = c

    def fit(self, X, y):
        """
        Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)
        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \coneleq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        # -a_i \leq 0
        # TODO(tulloch) - modify G, h so that we have a soft-margin classifier
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        # a_i \leq c
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        return np.ravel(solution['x'])

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = lagrange_multipliers > 1e-5

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self._kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel.calculate(x_i, x_j)
        return K


class SVMPredictor:
    def __init__(
            self,
            kernel: Kernel,
            bias, weights,
            support_vectors,
            support_vector_labels
    ):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel.calculate(x_i, x)
        return np.sign(result).item()


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return np.asarray(a)[p], np.asarray(b)[p]


def read_test_file():
    features = []
    classes = []
    with open('../lab01/dataset.txt') as f:
        for line in f:
            values = line.split(',')
            features.append([float(values[0]), float(values[1])])
            classes.append(float(1 if int(values[2]) == 1 else -1))  # may be change to float cast

    return unison_shuffled_copies(np.asarray(features), np.asarray(classes))


def get_data_sets(train, test, fold_count: int = 5):
    all_data = []
    all_data.extend(train)
    all_data.extend(test)
    return all_data[len(all_data) // fold_count:], all_data[0: len(all_data) // fold_count]


def test(
        dataset,
        fold_count: int = 5
):
    f_mera = FMeraCalculator([0, 1])

    train_x = dataset[0]
    train_y = dataset[1]

    test_x = []
    test_y = []
    for i in range(len(dataset[0]) // fold_count):
        train_x, test_x = get_data_sets(train_x, test_x, fold_count)
        train_y, test_y = get_data_sets(train_y, test_y, fold_count)

        print()
        print("Train {}".format(i))
        svm = SVMTrainer().fit(np.asarray(train_x), np.asarray(train_y))

        for features, true_cat in zip(test_x, test_y):
            predict_cat = svm.predict(features)
            print("pred {} true {}".format(predict_cat, true_cat))
            f_mera.add_data((predict_cat + 1) // 2, (true_cat + 1) // 2)

    return f_mera.get_mera()


if __name__ == '__main__':
    dataset = read_test_file()
    f_mera = test(dataset)
    print('Total F-Measure {}'.format(f_mera))
