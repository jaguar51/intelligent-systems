import math
import numpy as np

# http://www.machinelearning.ru/wiki/index.php?title=Коэффициент_корреляции_Пирсона
# http://statpsy.ru/pearson/formula-pirsona/
from collections import defaultdict


def pearson_correlation(x, y, k=None):
    correlation = {}

    for i in range(len(x)):
        correlation[i] = count_pearson_correlation_coefficient(x[i], y)

    correlation = sorted(correlation.items(), key=lambda d: abs(d[1]), reverse=True)

    if k is None:
        return correlation
    else:
        return correlation[0:k]


def count_pearson_correlation_coefficient(x, y):
    x_mean = x.mean()
    y_mean = y.mean()

    num = 0
    x_2 = 0
    y_2 = 0
    for x_el, y_el in zip(x, y):
        num += ((x_el - x_mean) * (y_el - y_mean))
        x_2 += (x_el - x_mean) ** 2
        y_2 += (y_el - y_mean) ** 2

    return num / math.sqrt(x_2 * y_2)


# http://www.machinelearning.ru/wiki/index.php?title=Коэффициент_корреляции_Спирмена
def spearman_correlation(x, y, k=None):
    correlation = {}

    rank_dict_y, ligament_y = get_rank_ligaments(y)
    for i in range(len(x)):
        correlation[i] = count_spearman_correlation_coefficient(x[i], y, rank_dict_y, ligament_y)

    correlation = sorted(correlation.items(), key=lambda d: abs(d[1]), reverse=True)

    if k is None:
        return correlation
    else:
        return correlation[0:k]


def count_spearman_correlation_coefficient(x, y, rank_dict_y, ligament_y):
    rank_dict_x, ligament_x = get_rank_ligaments(x)
    n = len(x)
    diff_rank = 0

    for i, x_el in enumerate(x):
        diff_rank += ((rank_dict_x[x_el] - (n + 1) / 2) * (rank_dict_y[y[i]] - (n + 1) / 2))

    coeff = diff_rank / (n * (n - 1) * (n + 1) - (ligament_x + ligament_y))
    return coeff


def define_rank(arr):
    rank_dict = {}
    count_dict = {}

    y = sorted(arr)
    for i, y_el in enumerate(y):
        rank_dict[y_el] = i
        if y_el not in count_dict:
            count_dict[y_el] = 0

        count_dict[y_el] += 1

    for key in count_dict.keys():
        if count_dict[key] != 1:
            rank_dict[key] /= count_dict[key]

    return rank_dict, count_dict


def get_rank_ligaments(arr):
    rank_dict, count_dict = define_rank(arr)
    ligaments = 0
    for key in count_dict.keys():
        ligaments += count_dict[key] * ((count_dict[key] ** 2) - 1)

    ligaments *= 1 / 2
    return rank_dict, ligaments


# https://habrahabr.ru/post/264915/
def info_gain_correlation(x, y, k=None):
    eps = 0.0001

    num_d = len(y)
    num_ck = {}
    num_fi_ck = {}
    num_nfi_ck = {}
    for xi, yi in zip(x, y):
        num_ck[yi] = num_ck.get(yi, 0) + 1
        for index, xii in enumerate(xi):
            if index not in num_fi_ck:
                num_fi_ck[index] = {}
                num_nfi_ck[index] = {}
            if not yi in num_fi_ck[index]:
                num_fi_ck[index][yi] = 0
                num_nfi_ck[index][yi] = 0
            if not xii == 0:
                num_fi_ck[index][yi] = num_fi_ck[index].get(yi) + 1
            else:
                num_nfi_ck[index][yi] = num_nfi_ck[index].get(yi) + 1

    num_fi = {}
    for fi, dic in num_fi_ck.items():
        num_fi[fi] = sum(dic.values())
    num_nfi = dict([(fi, num_d - num) for fi, num in num_fi.items()])
    HD = 0

    for ck, num in num_ck.items():
        p = float(num) / num_d
        HD = HD - p * math.log(p, 2)

    IG = {}
    for fi in num_fi_ck.keys():
        POS = 0
        for yi, num in num_fi_ck[fi].items():
            p = (float(num) + eps) / (num_fi[fi] + eps * len(dic))
            POS = POS - p * math.log(p, 2)

        NEG = 0
        for yi, num in num_nfi_ck[fi].items():
            p = (float(num) + eps) / (num_nfi[fi] + eps * len(dic))
            NEG = NEG - p * math.log(p, 2)
        p = float(num_fi[fi]) / num_d
        IG[fi] = round(HD - p * POS - (1 - p) * NEG, 4)

    IG = sorted(IG.items(), key=lambda d: d[1], reverse=True)

    if k is None:
        return IG
    else:
        return IG[0:k]
