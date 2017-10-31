from lab02.common import *


class GradientDescent:
    def __init__(self, alpha: float, iterations: int):
        self.theta: np.ndarray = np.asarray([])
        self.alpha: float = alpha
        self.iterations: int = iterations
        self.y_mean = 0
        self.y_std = 0
        self.calc_err_iter = 0

    def fit(self, x: np.ndarray, y: np.ndarray):
        theta: np.ndarray = np.ones(x.shape[1])

        self.y_mean = y.mean()
        self.y_std = y.std()
        y_normal = (y - self.y_mean) / self.y_std

        last_loss_f_res = None
        loss_f_res = None

        for i in range(self.iterations):
            self.calc_err_iter+=1
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y_normal

            alpha = 0.1
            if i > 0:
                loss_f_res = self.__error_internal__(x, y_normal, theta)
            if i > 1:
                alpha = self.chose_alpha(loss_f_res, last_loss_f_res)

            gradient = np.dot(x.transpose(), loss) / x.shape[0] * 2
            theta = theta - alpha * gradient

            last_loss_f_res = loss_f_res

        self.theta = theta

    def __error_internal__(self, x: np.ndarray, y: np.ndarray, weight):
        predict = np.dot(x, weight) * self.y_std + self.y_mean
        err = np.dot((y - predict).transpose(), (y - predict))
        return err / predict.shape[0]

    def chose_alpha(self, loss, last_loss):
        diff = np.abs(loss - last_loss)

        if diff < 10 :
            # return 1e6
            return 1_000_000

        if diff < 100:
            return 0.12

        if diff < 1000:
            return 0.01

        return 0.1

    def predict(self, x: np.ndarray):
        return np.dot(x.transpose(), self.theta) * self.y_std + self.y_mean


if __name__ == '__main__':
    x, y = read_split_data()
    x = (x - x.mean()) / x.std()

    g = GradientDescent(0.1, 5000)
    g.fit(x, y)

    compute_error_for_all(g, x, y)
    print(g.calc_err_iter)

    while False:
        raw_str = input("Input 'q' to exit or 3 number: \n")
        if raw_str.lower() == 'q':
            break

        arr = [int(num) for num in raw_str.replace(',', ' ').split(' ')]

        predict = g.predict(np.asarray(arr[:2]))
        error = compute_error(np.asarray([predict]), np.asarray([arr[2]]))
        print("Value = {}, Error = {}\n".format(predict, error))
