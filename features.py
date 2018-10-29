# This file contains all functions related to transforming and cleaning
# the features
import numpy as np

#   This dictionary contains, for each feature, the hand-picked minimum and
#   maximum values above / below which we decide that certain values are
#   considered to be outliers.
MIN_MAX_VALUES = {
    0:  (0,  500), 1 : (0,  220), 2: (0, 300),  3: (0, 320),
    5:  (0, 2300),                              8: (0, 100),  9: (0, 1000),
    10: (0,    6),                             13: (0, 150),
                   16: (0,  180),                            19: (0,  210),
                   21: (0, 1000),              23: (0, 500),
                   26: (0,  250),                            29: (0,  500),
}


def build_poly(x, degree):
    """
    Expand each feature of the x matrix into a polynomial of degree d. If d is negative,
    it will include all cross terms (i.e. x1 * x2). If d is positive, it will only include
    powers of individual features (x^1, x^2, ...).
    """

    if degree < 0:
        return build_poly_cross(x, -degree)

    return build_poly_single(x, degree)


def build_poly_single(x, degree):
    """
    Expand each feature of the x matrix individually into a polynomial of degree d
    """

    x_constant = np.ones(shape=(x.shape[0], 1))

    if degree == 0:
        return x_constant

    if degree == 1:
        return np.concatenate([x_constant, x], axis=1)

    xi = np.repeat(x, degree, axis=1)
    di = np.tile(np.arange(1, degree + 1), x.shape[1])

    x_expanded = np.apply_along_axis(lambda row: np.vectorize(lambda x, d: x ** d)(row, di), 1, xi)

    return np.concatenate([x_constant, x_expanded], axis=1)


def build_poly_cross(x, degree):
    """
    Expand features of the x matrix into a polynomial of degree d (including cross-products).
    Important note -  It only works for degree = 2 at the moment
    """

    cross_x = np.array([[row[i] * row[j] for i in range(len(row)) for j in range(len(row)) if j != i and j > i] for row in x])
    exp_x = build_poly_single(x, degree)

    return np.concatenate([exp_x, cross_x], axis=1)


def standardize(x):
    """Standardize a column x"""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


def standardize_all(x):
    """Standardize each column of x individually."""
    x = np.apply_along_axis(lambda xi: standardize(xi)[0], 0, x)
    return x


def normalize(x):
    """Normalize a column x"""
    return (x - x.min()) / (x.max() - x.min())


def normalize_all(x):
    """Standardize each column of x individually."""
    x = np.apply_along_axis(lambda xi: normalize(xi), 0, x)
    return x


def clamp(data, min_value, max_value, mode='clamp'):
    """Standardize each column of x individually."""

    if mode == 'clamp':
        data[data > max_value] = max_value
        data[data < min_value] = min_value
    elif mode == 'delete':
        data[data > max_value] = np.NaN
        data[data < min_value] = np.NaN
    else:
        raise AssertionError(f'Unknown mode {mode}')


def remove_errors(data):
    """Replace all instances of -999.0 with NaN."""
    data = data.copy()
    data[data == -999.0] = np.NaN
    return data


def remove_outliers(data, mode='clamp'):
    """Get rid of all outliers manually defined in MAX_MIN_VALUES."""
    data = data.copy()

    for key, value in MIN_MAX_VALUES.items():
        clamp(data[:, key], value[0], value[1], mode)

    return data


def remove_nan_features(x):
    """Remove columns of x that contain at least one NaN value."""
    x = x[:, ~np.any(np.isnan(x), axis=0)]
    return x


def remove_nan_samples(x, y):
    """Remove rows of x that contain at least one NaN value (and same for y)"""
    mask = np.any(np.isnan(x), axis=1)
    return x[~mask], y[~mask]


def decompose_categorical(x):
    """Create dummy variables for a given feature."""
    return np.array([np.vectorize(lambda xi: 1 if xi == i else 0)(x) for i in np.unique(x)]).T


def separate_features(x, indices):
    """Split dataset in two according to indices"""
    return x[:, indices], np.delete(x, indices, axis=1)



def split_data(y, x, ids):
    """
    Split the dataset into 5 hand-picked categories.

    - Category 0: All rows which do not have a value for 'DER_mass_MMC' (feature #0)
    - Category 1: All rows which have value 0 in 'PRI_jet_num' (feature #22)
    - Category 2: All rows which have value 1 in 'PRI_jet_num' (feature #22)
    - Category 3: All rows which have value 2 in 'PRI_jet_num' (feature #22)
    - Category 4: All rows which have value 3 in 'PRI_jet_num' (feature #22)

    """
    def categorize(x):
        if x[0] == -999.0:
            return 0
        else:
            return x[22] + 1

    categories = np.apply_along_axis(categorize, 1, x)

    xs = [x[categories == i] for i in np.arange(5)]
    ys = [y[categories == i] for i in np.arange(5)]
    ids = [ids[categories == i] for i in np.arange(5)]

    return ys, xs, ids


def clean_and_fit(clean, fit):

    def inner_function(y, x, h):
        y, x = clean(y, x, h)
        return fit(y, x, h)

    return inner_function


