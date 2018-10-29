# This file contains all functions that allow to cross-validate a model
# into training and testing subsets.
import numpy as np


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


#   Helpers
#   -------

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                    for k in range(k_fold)]
    return np.array(k_indices)

def cross_data(y, x, k_indices, k):

    # get k'th subgroup in test, others in train:
    x_tr = x[k_indices[np.arange(len(k_indices)) != k].flatten()]
    x_te = x[k_indices[k]]

    # Same for y
    y_tr = y[k_indices[np.arange(len(k_indices)) != k].flatten()]
    y_te = y[k_indices[k]]

    return x_tr, x_te, y_tr, y_te
