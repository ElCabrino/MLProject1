from cache import Cache
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from helpers import *
from features import *
from splits import *
from collections import defaultdict

class Model:

    def prepare(self, x, y, h):

        raise NotImplementedError

    def fit(self, x, y, h):

        raise NotImplementedError

    def test(self, x, y, w, h):

        raise NotImplementedError

    def evaluate_step(self, x, y, h):

        stored_res = self.cache.get(h)

        if stored_res != None:
            return dict(list(zip(stored_res.dtype.names, stored_res)))

        x, y = self.prepare(x, y, h)
        w = self.fit(x, y, h)
        res = self.test(x, y, w, h)
        self.cache.put(h, res)

        return { **h, **res }

    def evaluate(self, x, y, hs, filename):
        """Applies grid search algorithm to the cartesian product of the parameters
        passed in argument. If a 'file' is given, it will load from this file
        and write into it."""

        self.cache = Cache(filename)

        hs_items = sorted(hs.items())
        hs_keys = [hi[0] for hi in hs_items]
        hs_values = [hi[1] for hi in hs_items]

        (hs_grid) = np.meshgrid(*tuple(hs_values), indexing='ij')

        hs_pairs = [{}]
        for key in hs:
            temp = []
            for value in hs[key]:
                for pair in hs_pairs:
                    t = dict(pair)
                    t[key] = value
                    temp.append(t)
            hs_pairs = temp

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        return pool.starmap(self.fit_with_cache, [(x, y, h) for h in hs_pairs])

    def predict(self, h, x_tr, y_tr, name):

        x_tr, y_tr = self.prepare(x_tr, y_tr, h)
        w = self.fit(x_tr, y_tr, h)

        _, x_pred, ids = load_csv_data("data/test.csv", sub_sample=False)
        x_pred, _ = self.prepare(x_pred, None, h)
        y_pred = predict_labels(w, x_pred)

        create_csv_submission(ids, y_pred, name)

def plot_heatmap(res, hs, value, x, y):
    val = np.vectorize(lambda x: x[value])(res)

    index = 0

    for key in sorted(hs.keys()):
        if key == x or key == y:
            index = index + 1
        else:
            val = np.apply_along_axis(np.mean, index, val)

    ax = plt.imshow(1 / val, cmap='hot', interpolation='none')
    plt.show()

def find_arg_min(res, value):
    val = np.vectorize(lambda x: x[value])(res)
    index = np.where(val == val.min())

    return res[tuple([i[0] for i in index])]

class CrossValidationModel(Model):

    def __init__(self, model):

        self.model = model

    def prepare(self, x, y, h):

        return self.model.prepare(x, y, h)

    def fit(self, x, y, h):

        k_fold = int(h['k_fold'])
        seed = int(h['seed'])

        # Split data in k fold
        k_indices = build_k_indices(y, k_fold, seed)

        # We will store average over the k_fold in this
        averages = defaultdict(float)

        ws = []

        for k in range(0, k_fold):

            # Get split data
            x_tr, x_te, y_tr, y_te = cross_data(y, x, k_indices, k)

            # Perform fit on partitioned data
            w = self.model.fit(x_tr, y_tr, h)

            # Append everything so we can pass it to test
            ws.append(w)

        return ws

    def test(self, x, y, w, h):

        k_fold = int(h['k_fold'])
        seed = int(h['seed'])

        # Split data in k fold
        k_indices = build_k_indices(y, k_fold, seed)

        averages = defaultdict(int)

        for k in range(0, k_fold):

            x_tr, x_te, y_tr, y_te = cross_data(y, x, k_indices, k)

            errors_tr = self.model.test(x_tr, y_tr, w[k], h)
            errors_te = self.model.test(x_te, y_te, w[k], h)

            errors = {**{ k + '_tr': v for k, v in errors_tr.items() },
                      **{ k + '_te': v for k, v in errors_te.items() }}

            for key, error in errors.items():
                averages[key] += error / k_fold

        return averages

    def predict(self, h, x_tr, y_tr, name):

        return self.model.predict(h, x_tr, y_tr, name)
