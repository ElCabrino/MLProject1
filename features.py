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

    if degree < 0:
        return build_poly_cross(x, -degree)

    return build_poly_single(x, degree)

def build_poly_single(x, degree):

    x_constant = np.ones(shape=(x.shape[0], 1))

    if degree == 0:
        return x_constant

    if degree == 1:
        return np.concatenate([x_constant, x], axis=1)

    xi = np.repeat(x, degree, axis=1)
    di = np.tile(np.arange(1, degree + 1), x.shape[1])

    x_expanded = np.apply_along_axis(lambda row: np.vectorize(lambda x, d: x ** d)(row, di), 1, xi)

    return np.concatenate([x_constant, x_expanded], axis=1)

def build_poly_cross(x):

    cross_x = np.array([[row[i] * row[j] for i in range(len(row)) for j in range(len(row)) if j != i and j > i] for row in x])
    exp_x = build_poly_single(x, 2)

    return np.concatenate([exp_x, cross_x], axis=1)

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

def decompose_categorical(x):
    return np.array([np.vectorize(lambda xi: 1 if xi == i else 0)(x) for i in np.unique(x)]).T

def separate_features(x, indexes):
    return x[:, indexes], np.delete(x, indexes, axis=1)

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def normalize_all(x):
    x = np.apply_along_axis(lambda xi: normalize(xi), 0, x)
    return x
