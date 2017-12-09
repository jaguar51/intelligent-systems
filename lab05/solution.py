import numpy as np
from matplotlib import pyplot as plt
from matplotlib_venn import venn3

from lab05.fs_algorithms import pearson_correlation, spearman_correlation, info_gain_correlation


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

    return np.asarray(features), np.asarray(labels)


def pearson_example(features, labels, k=10):
    print("Pearson:")
    keys = []
    gain = pearson_correlation(features.T, labels, k)
    for k, v in gain:
        keys.append(k)
        print(k, v)

    print("---------------------")
    print()
    return keys


def spearman_example(features, labels, k=10):
    print("Spearman:")
    keys = []
    gain = spearman_correlation(features.T, labels, k)
    for k, v in gain:
        keys.append(k)
        print(k, v)

    print("---------------------")
    print()
    return keys


def ig_example(features, labels, k=10):
    print("IG:")
    keys = []
    gain = info_gain_correlation(features, labels, k)
    for k, v in gain:
        keys.append(k)
        print(k, v)

    print("---------------------")
    print()
    return keys


def get_common_elements(left_arr, right_arr):
    s_l = set(left_arr)
    s_r = set(right_arr)

    common = []

    for el in s_l:
        if el in s_r:
            common.append(el)

    return common


def plot_graphic(pearson, spearman, ig):
    s = (
        len(pearson),  # Pearson
        len(spearman),  # Spearman
        len(get_common_elements(pearson, ig)),  # Pearson+IG
        len(ig),  # IG
        len(get_common_elements(pearson, spearman)),  # Pearson+Spearman
        len(get_common_elements(get_common_elements(pearson, spearman), ig)),  # Pearson+Spearman+IG
        len(get_common_elements(spearman, ig)),  # Spearman+IG
    )

    v = venn3(subsets=s, set_labels=('Pearson', 'Spearman', 'IG'))
    plt.show()


if __name__ == '__main__':
    features, labels = read_data()
    k = 20
    pearson = pearson_example(features, labels, k)
    spearman = spearman_example(features, labels, k)
    ig = ig_example(features, labels, k)
    plot_graphic(pearson, spearman, ig)
