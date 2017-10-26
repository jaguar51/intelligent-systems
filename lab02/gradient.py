import math
from lab02.common import *


# z = ax + by + c // a, b, c
def compute_error_for_line_given_points(a, b, c, points):
    error = 0
    for row in points:
        x, y, z = row
        error += (z - calculate_value(a, b, c, x, y)) ** 2
    return math.sqrt(error / len(points))


def calculate_value(a, b, c, x, y):
    return a * x + b * y + c


def step_gradient(a_cur, b_cur, c_cur, points, learning_rate):
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0

    N = len(points)

    for i in range(0, len(points)):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]

        a_gradient += (2 / N) * ((a_cur * x + b_cur * y + c_cur) - z) * x
        b_gradient += (2 / N) * ((a_cur * x + b_cur * y + c_cur) - z) * y
        c_gradient += (2 / N) * ((a_cur * x + b_cur * y + c_cur) - z)

    new_a = a_cur - (learning_rate * a_gradient)
    new_b = b_cur - (learning_rate * b_gradient)
    new_c = c_cur - (learning_rate * c_gradient)

    return [new_a, new_b, new_c]


def gradient_descent(data, starting_a, starting_b, starting_c, learning_rate, iteration_count):
    a = starting_a
    b = starting_b
    c = starting_c
    for i in range(iteration_count):
        a, b, c = step_gradient(a, b, c, data, learning_rate)
        print("After {0} iterations a = {1}, b = {2}, c = {3}, error = {4}".format(
            i,
            a, b, c,
            compute_error_for_line_given_points(a, b, c, data)
        ))
    return [a, b, c]


if __name__ == '__main__':
    data = read_data()
    learning_rate = 0.0000002  # learning_rate = k / t, k - number constant, t - iteration num
    # initial_a = 176
    # initial_b = 19
    # initial_c = 6
    initial_a = 0
    initial_b = 0
    initial_c = 0
    num_iterations = 5000

    print("Starting gradient descent at a = {0}, b = {1}, c = {2}, error = {3}".format(
        initial_a,
        initial_b,
        initial_c,
        compute_error_for_line_given_points(initial_a, initial_b, initial_c, data)
    ))
    print("Running...")
    [a, b, c] = gradient_descent(data, initial_a, initial_b, initial_c, learning_rate, num_iterations)
    while False:
        sym = input()
        if sym == 'q':
            exit()

        x = int(sym)
        y = int(input())
        z = int(input())
        print("Value = {}, Error = {}".format(
            calculate_value(a, b, c, x, y),
            compute_error_for_line_given_points(a, b, c, [[x, y, z]]))
        )
