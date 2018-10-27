# This file contains all functions that allow to split a model
# into training and testing subsets.
import numpy as np

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

def batch_iter(y, tx, batch_size, num_batches=1, seed=0):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    np.random.seed(seed)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
