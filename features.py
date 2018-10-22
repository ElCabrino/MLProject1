# This file contains all functions related to transforming and cleaning
# the features

import numpy as np

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    return np.array([np.concatenate([[1.0]] + [[xi ** d for d in range(1, degree+1)] for xi in row]) for row in x])

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def clamp(data, max_value):
    data[data > max_value] = max_value

def remove_outliers(data):
    data = data.copy()
    clamp(data[:, 0], 500)
    clamp(data[:, 1], 220)
    clamp(data[:, 2], 300)
    clamp(data[:, 3], 320)
    # 4
    clamp(data[:, 5], 2300)
    # 6
    # 7
    clamp(data[:, 8], 100)
    clamp(data[:, 9], 1000)
    clamp(data[:, 10], 6)
    # 11
    # 12
    clamp(data[:, 13], 150)
    # 14
    # 15
    clamp(data[:, 16], 180)
    # 17
    # 18
    clamp(data[:, 19], 210)
    # 20
    clamp(data[:, 21], 1000)
    # 22
    clamp(data[:, 23], 500)
    # 24
    # 25
    clamp(data[:, 26], 250)
    # 27
    # 28
    clamp(data[:, 29], 500)

    return data

def clean_data(x):

    # Copy so that we don't touch the original data
    x = x.copy()

    # Replace wrong values
    x[x == -999.0] = np.NaN

    x = np.apply_along_axis(lambda xi: standardize(xi)[0], 0, x)
    x = x[:, ~np.all(np.isnan(x), axis=0)]

    return x
