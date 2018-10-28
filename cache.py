import os.path
import csv
import numpy as np


class Cache:

    def __init__(self, filename):

        self.filename = filename + '.csv'

        if os.path.isfile(self.filename):
            self.genArray()
            self.headers = self.array.dtype.names
        else: # Delay creation of array
            self.array = None

    def genArray(self):
        self.array = np.genfromtxt(self.filename, delimiter=',', names=True, case_sensitive=True, dtype=None)

    def headers(self):
        return self.hyperparams_names + self.values_names

    def contains(self, hyperparams):

        return self.get(hyperparams) != None

    def get(self, hyperparams):

        if self.array == None:
            return None

        res = self.array[np.bitwise_and.reduce([self.array[k] == v for k, v in hyperparams.items()])]

        if res.shape[0] == 0:
            return None
        else:
            return res

    def put(self, hyperparams, values):

        if self.contains(hyperparams):
            # For now, just avoid duplicates
            return

        self.headers = sorted(hyperparams.keys()) + sorted(values.keys())

        if not os.path.isfile(self.filename):
            with open(self.filename, 'w+') as fp:
                fp.write(','.join(self.headers) + '\n')
            self.genArray()

        allValues = {**hyperparams, **values}
        row = ','.join([str(allValues[k]) for k in self.headers]) + '\n'

        with open(self.filename,'a') as fd:
            fd.write(row)
