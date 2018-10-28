from cache import Cache
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from helpers import *
from features import *
from splits import *
from collections import defaultdict
from gradients import *


def stochastic_gradient_descent_e(gradient, loss, log=True):

    def inner_function(y, x, h, cache=None):

        seed = int(h['seed'])
        batch_size = int(h['batch_size'])
        num_batches = int(h['num_batches'])
        max_iters = int(h['max_iters'])
        gamma = float(h['gamma'])

        w = np.zeros(x.shape[1])
        seed_iter = seed

        err = {}

        for step in range(max_iters):

            for y_batch, x_batch in batch_iter(y, x, batch_size=batch_size, num_batches=num_batches, seed=seed_iter):

                # Compute gradient using the inner model
                grad = gradient(y_batch, x_batch, w, h)

                # grad = results['grad']
                # err = results['err']

                w = w - gamma * grad

            if step % 50 == 0:

                err = loss(y, x, w)

                if log:
                    print(f'iteration {step} - err = {err}')

            seed_iter += 1

        return {
            **h,
            **err,
            'w': w
        }

    return inner_function

def cross_validate(fit, validate):

    def fit_inner(y, x, h):

        k_fold = int(h['k_fold'])
        seed = int(h['seed'])

        # Split data in k fold
        k_indices = build_k_indices(y, k_fold, seed)

        # We will store average over the k_fold in this
        averages = defaultdict(float)

        weight_averages = np.zeros(x.shape[1])

        for k in range(0, k_fold):

            # Get split data
            x_tr, x_te, y_tr, y_te = cross_data(y, x, k_indices, k)

            # Perform fit on partitioned data
            result_tr = fit(y_tr, x_tr, h)

            # Extract weights from results
            w = result_tr['w']
            del result_tr['w']

            # Validate on test set
            result_te = validate(y_te, x_te, w)

            weight_averages = weight_averages + (w / k_fold)

            for key, value in result_tr.items():
                averages['avg_' + key + '_tr'] += value / k_fold

            for key, value in result_te.items():
                averages['avg_' + key + '_te'] += value / k_fold

        return { **h, **averages, 'w': weight_averages }

    return fit_inner


def clean_and_fit(clean, fit):

    def inner_function(y, x, h):
        y, x = clean(y, x, h)
        return fit(y, x, h)

    return inner_function


def fit_with_cache(fit, cache):
    """
    This function takes a fit function and looks in the cache to see if the
    parameters have already been evaluated. If so, it will return the values
    associated with the parameters.
    """

    if cache == None:
        return fit

    def fit_inner(y, x, h):

        stored_res = cache.get(h)

        # If there is a stored result, we simply take it.
        if stored_res != None:
            res = dict(list(zip(stored_res.dtype.names, *stored_res)))
            res['w'] = decode_w(res['w'])
            return res

        # Otherwise, we recompute
        result = fit(y, x, h)

        result_to_cache = { **result }
        result_to_cache['w'] = encode_w(result_to_cache['w'])

        cache.put(h, result_to_cache)

        return { **h, **result }

    return fit_inner

def evaluate(clean, fit, y, x, hs, cache):
    """Applies grid search algorithm to the cartesian product of the parameters
    passed in argument. If a 'file' is given, it will load from this file
    and write into it."""

    if cache != None:
        cache = Cache(cache)
    else:
        cache = None

    hs_items = sorted(hs.items())
    hs_keys = [hi[0] for hi in hs_items]
    hs_values = [hi[1] for hi in hs_items]

    (hs_grid) = np.meshgrid(*tuple(hs_values), indexing='ij')
    hs_params = np.vectorize(lambda *a: { hs_keys[i]: a[i] for i in range(len(a)) })(*tuple(hs_grid))
    hs_params = [(y, x, h) for h in hs_params.flat]

    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # res = pool.starmap(self.execute, hs_params)

    return [clean_and_fit(clean, fit_with_cache(fit, cache))(*params) for params in hs_params]

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
    h = res[tuple([i[0] for i in index])]
    return h
