from cache import Cache
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from helpers import *
from features import *
from splits import *
from collections import defaultdict
from gradients import *



def descent_with_cache(descent, round_size, cache, multiple=True, log=True):

    def find_last_result(cache, round_size, h):

        max_iters = int(h['max_iters'])
        h = { **h }

        while max_iters > 0:
            h['max_iters'] = max_iters
            result = cache.get(h)
            if result != None:
                return result
            else:
                max_iters = max_iters - round_size

        return None

    def inner_function(y, x, h):

        max_iters = int(h['max_iters'])
        seed = int(h['seed'])
        batch_size = int(h['batch_size'])
        num_batches = int(h['num_batches'])
        max_iters = int(h['max_iters'])
        gamma = float(h['gamma'])

        if max_iters % round_size != 0:
            raise InvalidArgumentException('Please make sure max_iter is a product of round_size')

        last_result = find_last_result(cache, round_size, h)

        start_n_iter = 0
        initial_w = None

        if last_result != None:

            start_n_iter = last_result['max_iters']

            if multiple:
                initial_w = decode_ws(last_result)
            else:
                initial_w = decode_w(last_result['w'])

        if start_n_iter == max_iters:
            return last_result

        number_of_rounds = (max_iters - start_n_iter) // round_size

        for round in range(0, number_of_rounds):

            n_iter = start_n_iter + round * round_size
            seed_iter = seed + n_iter

            modified_h = { **h }
            modified_h['max_iters'] = round_size
            modified_h['seed'] = seed_iter

            result = descent(y, x, modified_h, initial_w)

            if multiple:
                initial_w = extract_ws(result)
            else:
                initial_w = result['w']

            modified_h['max_iters'] = n_iter + round_size
            modified_h['seed'] = seed

            if multiple:
                result = encode_ws(result)

            result['w'] = encode_w(result['w'])

            cache.put(modified_h, result)

            if log:
                print(f'iteration {n_iter + round_size} - {remove_ws(result)}')

    return inner_function

def descent_with_loss(gradient, loss):

    def inner_function(y, x, h, initial_w):

        result = gradient(y, x, h, initial_w)
        err = loss(y, x, result['w'], h)

        return { **result, **err }

    return inner_function

def stochastic_gradient_descent_e(gradient):

    def inner_function(y, x, h, initial_w):

        seed = int(h['seed'])
        batch_size = int(h['batch_size'])
        num_batches = int(h['num_batches'])
        max_iters = int(h['max_iters'])
        gamma = float(h['gamma'])

        if (not (type(initial_w) is np.ndarray)) and initial_w is None:
            initial_w = np.zeros(x.shape[1])

        w = initial_w

        seed_iter = seed

        err = {}

        for step in range(max_iters):

            for y_batch, x_batch in batch_iter(y, x, batch_size=batch_size, num_batches=num_batches, seed=seed_iter):

                # Compute gradient using the inner model
                grad = gradient(y_batch, x_batch, w, h)
                w = w - gamma * grad

        return {
            **h,
            'w': w
        }

    return inner_function


def cross_validate_descent(descent, validate):

    def inner_function(y, x, h, initial_w):

        k_fold = int(h['k_fold'])
        seed_cv = int(h['seed_cv'])

        # Split data in k fold
        k_indices = build_k_indices(y, k_fold, seed_cv)

        # We will store average over the k_fold in this
        averages = defaultdict(float)

        weights = []

        if initial_w == None:
            initial_w = [ None for k in range(0, k_fold) ]

        for k in range(0, k_fold):

            # Get split data
            x_tr, x_te, y_tr, y_te = cross_data(y, x, k_indices, k)

            # Perform descent on partitioned data
            result_tr = descent(y_tr, x_tr, h, initial_w[k])

            # Extract weights from results
            w = result_tr['w']

            # Validate on test set
            validate_tr = validate(y_tr, x_tr, w, h)
            validate_te = validate(y_te, x_te, w, h)

            weights.append(w)

            for key, value in remove_h(validate_tr, h).items():
                averages['avg_' + key + '_tr'] += value / k_fold

            for key, value in remove_h(validate_te, h).items():
                averages['avg_' + key + '_te'] += value / k_fold

        weights_dict = { f'w_{k}': weights[k] for k in range(0, k_fold) }

        return {
            **h,
            **averages,
            'w': np.mean(weights, axis=0),
            **weights_dict
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
            # result_tr = remove_ws(result_tr)

            # Validate on test set
            validate_tr = validate(y_tr, x_tr, w, h)
            validate_te = validate(y_te, x_te, w, h)

            weight_averages = weight_averages + (w / k_fold)

            for key, value in validate_tr.items():
                averages['avg_' + key + '_tr'] += value / k_fold

            for key, value in validate_te.items():
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
            stored_res['w'] = decode_w(stored_res['w'])
            return stored_res

        # Otherwise, we recompute
        result = fit(y, x, h)

        result_to_cache = { **result }
        result_to_cache['w'] = encode_w(result_to_cache['w'])

        cache.put(h, result_to_cache)

        return { **h, **result }

    return fit_inner

def evaluate(clean, fit, y, x, hs):
    """Applies grid search algorithm to the cartesian product of the parameters
    passed in argument. If a 'file' is given, it will load from this file
    and write into it."""

    hs_items = sorted(hs.items())
    hs_keys = [hi[0] for hi in hs_items]
    hs_values = [hi[1] for hi in hs_items]

    (hs_grid) = np.meshgrid(*tuple(hs_values), indexing='ij')
    hs_params = np.vectorize(lambda *a: { hs_keys[i]: a[i] for i in range(len(a)) })(*tuple(hs_grid))
    hs_params = [(y, x, h) for h in hs_params.flat]

    # pool = multiprocessing.Pool(multiprocessing.cpu_count())
    # res = pool.starmap(self.execute, hs_params)

    return [clean_and_fit(clean, fit)(*params) for params in hs_params]

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
