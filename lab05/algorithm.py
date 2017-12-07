import math

# https://habrahabr.ru/post/264915/
# https://gist.github.com/iamaziz/02491e36490eb05a30f8
# https://stackoverflow.com/questions/25462407/fast-information-gain-computation
import numpy as np
from itertools import combinations


def info_gain(x, y, k=None, eps=0.0001):
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


# http://statpsy.ru/pearson/formula-pirsona/
# https://github.com/Millyzfw/pearson/blob/master/function.py
# https://github.com/dnase/correlation/blob/master/correlation.py
def pearson_for_one_element(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: len(n) * sum(map(lambda i: i ** 2, n)) - (sum(n) ** 2)
    return (len(x) * sum(map(lambda a: a[0] * a[1], zip(x, y))) - sum(x) * sum(y)) / math.sqrt(q(x) * q(y))


def pearson(x, y, k=None):
    correlation = {}

    for i in range(len(x[0])):
        print(i)
        correlation[i] = pearson_for_one_element(np.asarray(x)[:, i], y)

    correlation = sorted(correlation.items(), key=lambda d: d[1], reverse=True)

    if k is None:
        return correlation
    else:
        return correlation[0:k]


def spearman_for_one_element(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))


def spearman(x, y, k=None):
    correlation = {}

    for i in range(len(x[0])):
        correlation[i] = spearman_for_one_element(np.asarray(x)[:, i], y)

    correlation = sorted(correlation.items(), key=lambda d: d[1], reverse=True)

    if k is None:
        return correlation
    else:
        return correlation[0:k]


def kendall_for_one_element(x, y):
    assert len(x) == len(y) > 0
    c = 0  # concordant count
    d = 0  # discordant count
    t = 0  # tied count
    for (i, j) in combinations(range(len(x)), 2):
        s = (x[i] - x[j]) * (y[i] - y[j])
        if s:
            c += 1
            d += 1
            if s > 0:
                t += 1
            elif s < 0:
                t -= 1
        else:
            if x[i] - x[j]:
                c += 1
            elif y[i] - y[j]:
                d += 1
    return t / math.sqrt(c * d)


def kendall(x, y, k=None):
    correlation = {}

    for i in range(len(x[0])):
        correlation[i] = kendall_for_one_element(np.asarray(x)[:, i], y)

    correlation = sorted(correlation.items(), key=lambda d: d[1], reverse=True)

    if k is None:
        return correlation
    else:
        return correlation[0:k]
