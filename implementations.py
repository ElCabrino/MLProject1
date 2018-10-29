import numpy as np

from costs import *


#   Required implementations
#   -----------------------
#
#   Below are the specific functions required to be implemented as part of
#   the assignment. Because of the nature of our code, they often ...

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    h = {'max_iters': max_iters, 'gamma': gamma}

    results = descent_with_loss(
        gradient_descent_e(least_squares_gradient),
        compute_mse
    )(y, tx, h, initial_w)

    return results['w'], results['mse']


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    h = {'max_iters': max_iters, 'gamma': gamma, 'seed': 0, 'batch_size': 1}

    results = descent_with_loss(
        stochastic_gradient_descent_e(least_squares_gradient),
        compute_mse
    )(y, tx, h, initial_w)

    return results['w'], results['mse']


def least_squares(y, tx):
    """calculate the least squares (analytical solution)."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression (analytical solution)."""

    t = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])

    a = tx.T.dot(tx) + t
    b = tx.T.dot(y)

    return np.linalg.solve(a, b)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implement logistic regression with SG."""

    h = {'max_iters': max_iters, 'gamma': gamma}

    results = descent_with_loss(
        gradient_descent_e(logistic_gradient),
        compute_logistic_error
    )(y, tx, h, initial_w)

    return results['w'], results['logistic_err']


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """implement logistic regression with SG and Ridge Regularization"""

    h = {'max_iters': max_iters, 'gamma': gamma, 'lambda': lambda_ }

    results = descent_with_loss(
        gradient_descent_e(logistic_gradient_ridge),
        compute_logistic_error
    )(y, tx, h, initial_w)

    return results['w'], results['logistic_err']


#   Gradient functions
#   ------------------
#
#   These are higher-level functions that can be wrapped around
#   lower-level descent functions such as computing the gradient.

def least_squares_gradient(y, x, w, h={}):
    """Compute the gradient."""
    err = y - x.dot(w)
    grad = -x.T.dot(err) / len(err)
    return grad, err


def logistic_gradient(y, x, w, h={}):
    """Compute gradient for the logistic regression algorithm"""

    return x.T @ (logistic_function(x @ w) - y) / y.shape[0]


#   Regularized gradient functions
#   ------------------------------
#
#   These are the same as above but with an additional regularization term.

def least_squares_gradient_lasso(y, x, w, h={}):
    return least_squares_gradient(y, x, w) + float(h['lambda']) * np.sign(w)


def least_squares_gradient_ridge(y, x, w, h={}):
    return least_squares_gradient(y, x, w) + float(h['lambda']) * w


def logistic_gradient_lasso(y, x, w, h):
    return logistic_gradient(y, x, w) + float(h['lambda']) * np.sign(w)


def logistic_gradient_ridge(y, x, w, h):
    return logistic_gradient(y, x, w) + float(h['lambda']) * w


#   Descent wrappers
#   ----------------
#
#   These are higher-level functions that can be wrapped around
#   gradient-computing function to run a full gradient descent (stochastic
#   or normal)

def stochastic_gradient_descent_e(gradient):
    def inner_function(y, x, h, initial_w):

        seed = int(h['seed'])
        batch_size = int(h['batch_size'])
        num_batches = int(h['num_batches'])
        max_iters = int(h['max_iters'])
        gamma = float(h['gamma'])

        if (not (type(initial_w) is np.ndarray)) and initial_w is None:
            initial_w = np.zeros(x.shape[1])

        w = initial_w

        seed_iter = seed

        err = {}

        for step in range(max_iters):

            for y_batch, x_batch in batch_iter(y, x, batch_size=batch_size, num_batches=num_batches, seed=seed_iter):
                # Compute gradient using the inner model
                grad = gradient(y_batch, x_batch, w, h)
                w = w - gamma * grad

        return {
            **h,
            'w': w
        }

    return inner_function


def gradient_descent_e(gradient):

    def inner_function(y, x, h, initial_w):

        max_iters = int(h['max_iters'])
        gamma = float(h['gamma'])

        if (not (type(initial_w) is np.ndarray)) and initial_w is None:
            initial_w = np.zeros(x.shape[1])

        w = initial_w

        for step in range(max_iters):
            grad = gradient(y, x, w, h)
            w = w - gamma * grad

        return {
            **h,
            'w': w
        }

    return inner_function


def descent_with_loss(gradient, loss):
    """
    Wrap around a gradient descent function and calculates the final
    loss according to the weights found and loss function provided.
    This was separated from the descent itself to make it easier to log
    intermediate steps of a descent into the cache and give flexibility
    as to which errors should be logged.
    """

    def inner_function(y, x, h, initial_w):
        result = gradient(y, x, h, initial_w)
        err = loss(y, x, result['w'], h)

        return {**result, **err}

    return inner_function


#   Advanced Logistic Regression
#   ----------------------------
#
#   These methods could be used for second degree logistic regression
#   but we have not used them in our code.


# def compute_S(tx, w):
#     """"Compute S matrix for second order logistic regression"""
#     n = tx.shape[0]
#     S = np.zeros([n, n])
#     for i in range(n):
#         sigma_xW = logistic_function(tx[i].T @ w)
#         S[i, i] = sigma_xW * (1 - sigma_xW)
#     return S
#
#
# def compute_H(tx, w):
#     """Compute H matrix for second order logistic regression"""
#     S = compute_S(tx, w)
#     return tx.T @ S @ tx
#
#
# def newton_method(y, tx, initial_w, batch_size, max_iters, gamma):
#     "Second order Logistic Regression with SGD"
#     w = initial_w
#     for n_iter in range(max_iters):
#         for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
#             H = compute_H(tx_batch, w)
#             H_inv = np.linalg.inv(H)
#             grad = compute_logistic_gradient(y_batch, tx_batch, w)
#             w = w - gamma * H_inv @ grad
#     return w


#   Helpers
#   -------

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
