import numpy as np
from matplotlib import pyplot as plt
from matplotlib_venn import venn3

from lab05.fs_algorithms import pearson_correlation, spearman_correlation, info_gain_correlation
from utils.fmera import FMeraCalculator


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
        # print(k, v)

    return keys


def spearman_example(features, labels, k=10):
    print("Spearman:")
    keys = []
    gain = spearman_correlation(features.T, labels, k)
    for k, v in gain:
        keys.append(k)
        # print(k, v)

    return keys


def ig_example(features, labels, k=10):
    print("IG:")
    keys = []
    gain = info_gain_correlation(features, labels, k)
    for k, v in gain:
        keys.append(k)
        # print(k, v)

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


def test(features_train, labels_train, features_test):
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(features_train, labels_train)
    return clf.predict(features_test)


def test_with_mera(features_train, labels_train, features_test, labels_test):
    f_mera_calc = FMeraCalculator([-1, 1])
    predict_res = test(features_train, labels_train, features_test)

    for pred, true in zip(predict_res, labels_test):
        f_mera_calc.add_data(pred, true)

    print("F-Measure {}".format(f_mera_calc.get_mera()))
    print("---------------------")
    print()


def get_features_by_num(features, nums):
    new_features = []

    features_t = features.T
    for num in nums:
        new_features.append(features_t[num])

    return np.asarray(new_features).T


if __name__ == '__main__':
    features_train, labels_train = read_data('arcene_train')
    features_test, labels_test = read_data('arcene_valid')

    pearson = pearson_example(features_train, labels_train, 20)
    test_with_mera(
        get_features_by_num(features_train, pearson),
        labels_train,
        get_features_by_num(features_test, pearson),
        labels_test
    )

    spearman = spearman_example(features_train, labels_train, 100)
    test_with_mera(
        get_features_by_num(features_train, spearman),
        labels_train,
        get_features_by_num(features_test, spearman),
        labels_test
    )

    ig = ig_example(features_train, labels_train, 20)
    test_with_mera(
        get_features_by_num(features_train, ig),
        labels_train,
        get_features_by_num(features_test, ig),
        labels_test
    )
    plot_graphic(pearson, spearman, ig)
