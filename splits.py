# This file contains all functions that allow to split a model
# into training and testing subsets.

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

def cross_validation_step(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""

    # get split data
    x_tr, x_te, y_tr, y_te = cross_data(y, x, k_indices, k)

    # form data with polynomial degree:
    x_tr = build_poly(x_tr, degree)
    x_te = build_poly(x_te, degree)

    # ridge regression:
    w = ridge_regression(y_tr, x_tr, lambda_)

    # calculate the loss for train and test data:
    loss_tr = compute_mse(y_tr, x_tr, w)
    loss_te = compute_mse(y_te, x_te, w)

    return loss_tr, loss_te

def cross_validation(y, x, k_fold, lambda_, degree, seed):

    avg_mse_tr = 0
    avg_mse_te = 0

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    for k in range(0, k_fold):
        loss_tr, loss_te = cross_validation_step(y, x, k_indices, k, lambda_, degree)
        avg_mse_tr += loss_tr / k_fold
        avg_mse_te += loss_te / k_fold

    rmse_tr = np.sqrt(2 * avg_mse_tr)
    rmse_te = np.sqrt(2 * avg_mse_te)

    return rmse_tr, rmse_te
