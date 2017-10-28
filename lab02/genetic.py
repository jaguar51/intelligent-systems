import random

from lab02.common import *


class Genetic:
    def __init__(self, number_of_generations: int, population_size: int = 1):
        self.iterations: int = number_of_generations
        self.theta: np.ndarray = np.asarray([])
        self.population_size = population_size
        self.children = 10

    def fit(self, x: np.ndarray, y: np.ndarray):
        population = self.__generate_population__(self.population_size, x.shape[1])

        for i in range(self.iterations):
            offspring = []
            for genom in population:
                for j in range(self.children):
                    child = self.__mutation__(genom, 1)
                    child_loss = self.__error_internal__(x, y, child)
                    offspring.append([child_loss, child])

            population = self.__selection__(offspring, len(population))

        self.theta = population[0]

    def predict(self, x: np.ndarray):
        return self.__calculate_internal__(x, self.theta)

    def __error_internal__(self, x: np.ndarray, y: np.ndarray, weight):
        predict = self.__calculate_internal__(x, weight)
        err = np.dot((y - predict).transpose(), (y - predict))
        return err / predict.shape[0]

    def __calculate_internal__(self, x: np.ndarray, theta: np.ndarray):
        return np.dot(x, theta)

    def __generate_population__(self, p, w_size):
        population = []
        for i in range(p):
            model = []
            for j in range(w_size):
                model.append(2 * random.random() - 1)
            population.append(model)

        return np.array(population)

    def __mutation__(self, genom: np.ndarray, modify_val: float = 0.1, modify_probability: float = 0.5):
        mutant = []
        for gen in genom:
            if random.random() <= modify_probability:
                gen += modify_val * (2 * random.random() - 1)
            mutant.append(gen)
        return mutant

    def __selection__(self, offspring, population):
        offspring.sort()
        population = [kid[1] for kid in offspring[:population]]
        return population


if __name__ == '__main__':
    x, y = read_split_data()

    g = Genetic(5000)
    g.fit(x, y)

    compute_error_for_all(g, x, y)

    while False:
        raw_str = input("Input 'q' to exit or 3 number: \n")
        if raw_str.lower() == 'q':
            break

        arr = [int(num) for num in raw_str.replace(',', ' ').split(' ')]

        predict = g.predict(np.asarray(arr[:2]))
        print("Value = {}\n".format(predict))
