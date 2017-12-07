import numpy as np

from lab05.algorithm import *


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def read_data(filename='arcene_train'):
    data_filename = filename + '.data'
    labels_filename = filename + '.labels'

    labels = []
    features = []

    with open(labels_filename) as f:
        for line in f:
            labels.extend([int(num) for num in line.split(' ')])

    with open(data_filename) as f:
        for line in f:
            features.extend([int(num) for num in line.strip().split(' ')])

    features = list(chunks([num for num in features], len(features) // len(labels)))

    return features, labels


def ig_example(features, labels):
    print("IG:")
    gain = info_gain(features, labels, 10)
    for k, v in gain:
        print(k, v)

    print("---------------------")
    print()


def pearson_example(features, labels):
    print("Pearson:")
    gain = pearson(features, labels, 10)
    for k, v in gain:
        print(k, v)

    print("---------------------")
    print()


def spearman_example(features, labels):
    print("Spearman:")
    gain = spearman(features, labels, 10)
    for k, v in gain:
        print(k, v)

    print("---------------------")
    print()


def kendal_example(features, labels):
    print("Kendal:")
    gain = kendall(features, labels, 10)
    for k, v in gain:
        print(k, v)

    print("---------------------")
    print()


if __name__ == '__main__':
    features, labels = read_data()
    ig_example(features, labels)
    pearson_example(features, labels)
    spearman_example(features, labels)
    kendal_example(features, labels)
