import matplotlib.pyplot as plt

from features import *
from implementations import *


def evaluate(fit, y, x, hs):
    """Applies grid search algorithm to the cartesian product of the parameters
    passed in argument. If a 'file' is given, it will load from this file
    and write into it."""

    hs_items = sorted(hs.items())
    hs_keys = [hi[0] for hi in hs_items]
    hs_values = [hi[1] for hi in hs_items]

    (hs_grid) = np.meshgrid(*tuple(hs_values), indexing='ij')
    hs_params = np.vectorize(lambda *a: { hs_keys[i]: a[i] for i in range(len(a)) })(*tuple(hs_grid))
    hs_params = [(y, x, h) for h in hs_params.flat]

    return [fit(*params) for params in hs_params]


def plot_heatmap(res, hs, value, x, y):

    val = np.vectorize(lambda x: x[value])(res)

    index = 0

    for key in sorted(hs.keys()):
        if key == x or key == y:
            index = index + 1
        else:
            val = np.apply_along_axis(np.mean, index, val)
            # if key in filter_values.keys():
            #     find_match = lambda row: row[np.where(np.vectorize(lambda elem: elem[key] == filter_values[key])(row))[0]][0]
            #     val = np.apply_along_axis(find_match, index, val)
            # else:

    ax = plt.imshow(1 / val, cmap='hot', interpolation='none')
    plt.show()


def find_arg_min(res, value):
    val = np.vectorize(lambda x: x[value])(res)
    index = np.where(val == val.min())
    h = res[index]
    return h
