from cache import Cache
import numpy as np


class Model:

    def __init__(self, x, y):

        self.raw_x = x
        self.raw_y = y

    def prepare(self, x, y):

        raise NotImplementedError

    def fit(self, x, y, h):

        raise NotImplementedError

    def get_weights(self, h):

        x, y = self.prepare(self.raw_x, self.raw_y)
        _, w = self.fit(x, y, h)

        return w

    def fit_with_cache(self, x, y, h):

        stored_res = self.cache.get(h)

        if stored_res != None:
            # if log:
                # print(f'Parameters {h} already computed, skippin')
            return dict(list(zip(stored_res.dtype.names, stored_res)))

        res, _ = self.fit(x, y, h)
        self.cache.put(h, res)

        # if log:
        #     print(f'Finished computing {h}.')

        return { **h, **res }

    def evaluate(self, hs, filename):
        """Applies grid search algorithm to the cartesian product of the parameters
        passed in argument. If a 'file' is given, it will load from this file
        and write into it."""

        x, y = self.prepare(self.raw_x, self.raw_y)
        self.cache = Cache(filename)

        hs_items = sorted(hs.items())
        hs_keys = [hi[0] for hi in hs_items]
        hs_values = [hi[1] for hi in hs_items]

        (hs_grid) = np.meshgrid(*tuple(hs_values))
        return np.vectorize(lambda *a: self.fit_with_cache(x, y, { hs_keys[i]: a[i] for i in range(len(a)) }))(*tuple(hs_grid))
