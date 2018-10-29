import os.path
from helpers import *


class Cache:

    def __init__(self, filename):

        self.filename = filename + '.csv'

        if os.path.isfile(self.filename):
            self.generate_array()
            self.headers = self.array.dtype.names
        else: # Delay creation of array
            self.array = None

    def generate_array(self):
        self.array = np.genfromtxt(self.filename, delimiter=',', names=True, case_sensitive=True, dtype=None)

    def contains(self, hyperparams):

        return self.get(hyperparams) is not None

    def get(self, hyperparams):

        if self.array is None:
            return None

        res = self.array[np.bitwise_and.reduce([self.array[k] == v for k, v in hyperparams.items()])]

        if res.shape[0] == 0:
            return None
        else:
            res = dict(list(zip(res.dtype.names, *res)))
            return res

    def put(self, hyperparams, values):

        values = remove_h(values, hyperparams)

        if self.contains(hyperparams):
            # For now, just avoid duplicates
            return

        self.headers = sorted(hyperparams.keys()) + sorted(values.keys())

        if not os.path.isfile(self.filename):
            with open(self.filename, 'w+') as fp:
                fp.write(','.join(self.headers) + '\n')
            self.generate_array()

        all_values = { **hyperparams, **values }
        row = ','.join([str(all_values[k]) for k in self.headers]) + '\n'

        with open(self.filename, 'a') as fd:
            fd.write(row)


#   Wrapper functions
#   -----------------
#
#   These are functions that wrap around fit or descent functions and log
#   them in a cache. This avoids re-computing values which have been already
#   evaluated.


def descent_with_cache(descent, round_size, cache, multiple=True, log=True):

    def find_last_result(cache, round_size, h):

        max_iters = int(h['max_iters'])
        h = { **h }

        while max_iters > 0:
            h['max_iters'] = max_iters
            result = cache.get(h)
            if result is not  None:
                return result
            else:
                max_iters = max_iters - round_size

        return None

    def inner_function(y, x, h):

        seed = int(h['seed'])
        max_iters = int(h['max_iters'])

        if max_iters % round_size != 0:
            raise RuntimeError('Please make sure max_iter is a product of round_size')

        last_result = find_last_result(cache, round_size, h)

        start_n_iter = 0
        initial_w = None

        if last_result is not None:

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

            modified_h = {**h}
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


def fit_with_cache(fit, cache):
    """
    This function takes a fit function and looks in the cache to see if the
    parameters have already been evaluated. If so, it will return the values
    associated with the parameters.
    """

    if cache is None:
        return fit

    def fit_inner(y, x, h):

        stored_res = cache.get(h)

        # If there is a stored result, we simply take it.
        if stored_res is not None:
            stored_res['w'] = decode_w(stored_res['w'])
            return stored_res

        # Otherwise, we recompute
        result = fit(y, x, h)

        result_to_cache = { **result }
        result_to_cache['w'] = encode_w(result_to_cache['w'])

        cache.put(h, result_to_cache)

        return {**h, **result}

    return fit_inner


#   Other Helpers
#   -------------


def encode_w(w):
    return '|'.join(map(lambda wi: str(wi), w))


def decode_w(w):
    return np.array([float(x) for x in str(w)[2:-1].split('|')])


def encode_ws(d):

    d_mut = {**d}

    for key, value in sorted(d.items()):
        if re.match('w_\d+', key):
            d_mut[key] = encode_w(value)

    return d_mut


def decode_ws(d):

    return [decode_w(w) for w in extract_ws(d) ]


def extract_ws(d):

    array = []

    for key, value in sorted(d.items()):
        if re.match('w_\d+', key):
            array.append(value)

    return array


def remove_ws(d):

    d_mut = {**d}

    for key, _ in d.items():
        if re.match('^w', key):
            del d_mut[key]

    return d_mut


def remove_h(d, h):

    d_mut = {**d}

    for key, _ in h.items():
        if key in d:
            del d_mut[key]

    return d_mut
