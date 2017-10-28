from lab02.common import *
import numpy as np


class GradientDescent:
    def __init__(self, alpha: float, iterations: int):
        self.theta: np.ndarray = np.asarray([])
        self.alpha: float = alpha
        self.iterations: int = iterations

    def fit(self, x: np.ndarray, y: np.ndarray):
        theta: np.ndarray = np.ones(x.shape[1])

        for i in range(self.iterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y
            gradient = np.dot(x.transpose(), loss) / x.shape[0] * 2
            theta = theta - self.alpha * gradient

        self.theta = theta

    def predict(self, x: np.ndarray):
        return np.dot(x.transpose(), self.theta)


def compute_error(predict, y):
    predict_ = np.dot((y - predict).transpose(), (y - predict))
    # return np.sqrt(predict_ / predict.shape[0])
    return predict_ / predict.shape[0]


def compute_error_for_all(g: GradientDescent, x: np.ndarray, y: np.ndarray):
    pred = []
    for i in range(len(x)):
        g_predict = g.predict(x[i])
        pred.append(g_predict)

    pred = np.asarray(pred)

    error = compute_error(pred, y)

    print("Total error = {}\n".format(error))


if __name__ == '__main__':
    x, y = read_split_data()
    g = GradientDescent(0.0000002, 5000)
    g.fit(x, y)

    compute_error_for_all(g, x, y)

    while True:
        raw_str = input("Input 'q' to exit or 3 number: \n")
        if raw_str.lower() == 'q':
            break

        arr = [int(num) for num in raw_str.split(' ')]

        predict = g.predict(np.asarray(arr[:2]))
        error = compute_error(np.asarray([predict]), np.asarray([arr[2]]))
        print("Value = {}, Error = {}\n".format(predict, error))
