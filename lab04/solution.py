import cvxopt
import math

import numpy as np

from lab03.solution import FMeraCalculator


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

        svm = SVM().fit(np.asarray(train_x), np.asarray(train_y))

        for features, true_cat in zip(test_x, test_y):
            predict_cat = svm.predict(features)
            print("pred {} true {}".format(predict_cat, true_cat))
            f_mera.add_data((predict_cat + 1) // 2, (true_cat + 1)//2)

    return f_mera.get_mera()


def linear_kernel(x1, x2):
    n_x1 = [x1[0], x1[1], math.sqrt(x1[0]**2 + x1[1]**2)]
    n_x2 = [x2[0], x2[1], math.sqrt(x2[0]**2 + x2[1]**2)]
    return np.dot(n_x1, n_x2)
    # return np.dot(x1, x2)


# def polynomial_kernel(x, y, p=3):
#     return (1 + np.dot(x, y)) ** p


class SVM:
    def __init__(self, kernel=linear_kernel, C: float = 0.1):
        self.kernel = kernel
        self.C = C

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # params
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.vstack((
            np.diag(np.ones(n_samples) * -1),
            np.identity(n_samples)
        )))
        h = cvxopt.matrix(np.hstack((
            np.zeros(n_samples),
            np.ones(n_samples) * self.C
        )))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]  # indexes for True values
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("{} support vectors out of {} points".format(len(ind), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

        return self

    def predict(self, X):
        # if np.sign(self.project(X)) > 0:
        #     return 1
        # else:
        #     return 0
        #
        return np.sign(self.project(X))

    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b


if __name__ == '__main__':
    dataset = read_test_file()
    f_mera = test(dataset)
    print('Total F-Measure {}'.format(f_mera))
