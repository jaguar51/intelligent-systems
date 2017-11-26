import scipy.stats as stat
import numpy as np


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
            classes.append(float(1 if int(values[2]) == 1 else -1))

    return unison_shuffled_copies(np.asarray(features), np.asarray(classes))


def get_data_sets(train, test, fold_count: int = 5):
    all_data = []
    all_data.extend(train)
    all_data.extend(test)
    return all_data[len(all_data) // fold_count:], all_data[0: len(all_data) // fold_count]


def transform_for_knn(x, y):
    from lab01.point import Point

    data = []
    for x_el, y_el in zip(x, y):
        data.append(Point(x_el[0], x_el[1], 1 if int(y_el) == 1 else 0))

    return data


def get_best_knn(train):
    from lab01.solution import KNearesNeighbour
    from lab01.kernels import GausKernel
    from lab01.calculators import EuclidCalculator
    from lab01.calculators import ConusTransformer

    return KNearesNeighbour(
        kernel=GausKernel(),
        dist_calc=EuclidCalculator(ConusTransformer()),
        train=train
    )


def get_best_svm():
    from lab04.solution import SVMTrainer
    from lab04.solution import GaussianKernel
    from lab04.solution import ParabaloidTransformer

    return SVMTrainer(
        kernel=GaussianKernel(transformer=ParabaloidTransformer()),
        c=5
    )


def test(
        dataset,
        fold_count: int = 5
):
    train_x = dataset[0]
    train_y = dataset[1]

    test_x = []
    test_y = []

    total_knn_predict = []
    total_svm_predict = []

    for i in range(fold_count):
        train_x, test_x = get_data_sets(train_x, test_x, fold_count)
        train_y, test_y = get_data_sets(train_y, test_y, fold_count)

        knn = get_best_knn(transform_for_knn(train_x, train_y))
        svm = get_best_svm().fit(np.asarray(train_x), np.asarray(train_y))

        arr_left = []
        arr_right = []
        for features, true_cat in zip(test_x, test_y):
            knn_predict = knn.predict(transform_for_knn([features], [true_cat])[0]) * 2 - 1
            svm_predict = svm.predict(features)

            arr_left.append(knn_predict)
            arr_right.append(svm_predict)

        t, p = stat.wilcoxon(arr_left, arr_right)
        print('Wilcoxon: {}; p-value: {}'.format(t, p))

        total_knn_predict.extend(arr_left)
        total_svm_predict.extend(arr_right)

    return stat.wilcoxon(total_knn_predict, total_svm_predict)


if __name__ == '__main__':
    dataset = read_test_file()
    t, p = test(dataset)
    print()
    print()
    print('Wilcoxon: {}; p-value: {}'.format(t, p))
