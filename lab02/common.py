import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # dont remove
import numpy as np


def read_data(file_name: str = 'prices.txt'):
    with open(file_name, 'r') as file:
        reader = csv.reader(file)
        hader = next(reader)
        rows = [[float(row[0]), float(row[1]), float(row[2])] for row in reader]

    return rows


def read_split_data(file_name: str = 'prices.txt'):
    data = read_data(file_name)
    x = [[el[0], el[1]] for el in data]
    y = [el[2] for el in data]
    return np.asarray(x), np.asarray(y)


def plot_data(data):
    x = [el[0] for el in data]
    y = [el[1] for el in data]
    z = [el[2] for el in data]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z)
    plt.show()
