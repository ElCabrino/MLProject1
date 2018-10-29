import seaborn as sns

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


def plot_heatmap(res, value, x_axis, y_axis, max_value=float('inf')):

    def find_value(res, x_key, x_value, y_key, y_value, value):

        for result in res:
            # print(result)
            if result[x_key] == x_value and result[y_key] == y_value:
                return result[value]

        return None

    x_axis_values = np.unique(np.vectorize(lambda res: res[x_axis])(res))
    y_axis_values = np.unique(np.vectorize(lambda res: res[y_axis])(res))

    (grid) = np.meshgrid(x_axis_values, y_axis_values)

    heatmap_params = np.vectorize(lambda *a: [res, x_axis, a[0], y_axis, a[1], value])(*tuple(grid))
    heatmap = np.vectorize(lambda params: find_value(*params))(heatmap_params)
    heatmap = np.where(heatmap > max_value, max_value, heatmap)

    sns.heatmap(heatmap, cmap='hot_r', xticklabels=x_axis_values, yticklabels=y_axis_values)

def find_arg_min(res, value):
    val = np.vectorize(lambda x: x[value])(res)
    index = np.where(val == val.min())[0][0]
    h = res[index]
    return h
