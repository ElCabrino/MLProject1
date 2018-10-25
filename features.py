# This file contains all functions related to transforming and cleaning
# the features
import numpy as np

MIN_MAX_VALUES = {
    0:  (0,  500), 1 : (0,  220), 2: (0, 300),  3: (0, 320),
    5:  (0, 2300),                              8: (0, 100),  9: (0, 1000),
    10: (0,    6),                             13: (0, 150),
                   16: (0,  180),                            19: (0,  210),
                   21: (0, 1000),              23: (0, 500),
                   26: (0,  250),                            29: (0,  500),
}

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

def standardize_all(x):
    x = np.apply_along_axis(lambda xi: standardize(xi)[0], 0, x)
    return x

def clamp(data, min_value, max_value, mode='clamp'):
    if mode == 'clamp':
        data[data > max_value] = max_value
        data[data < min_value] = min_value
    else:
        data[data > max_value] = np.NaN
        data[data < min_value] = np.NaN

def remove_errors(data):
    data = data.copy()
    data[data == -999.0] = np.NaN
    return data

def remove_outliers(data, mode='clamp'):
    data = data.copy()

    for key, value in MIN_MAX_VALUES.items():
        clamp(data[:, key], value[0], value[1], mode)

    return data

def remove_nan_features(x):
    x = x[:, ~np.any(np.isnan(x), axis=0)]
    return x

def remove_nan_samples(x, y):
    mask = np.any(np.isnan(x), axis=1)
    return x[~mask], y[~mask]
