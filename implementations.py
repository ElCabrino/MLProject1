import numpy as np

def compute_mse(y, tx, w):
    e = y - tx@w
    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def logistic_function(x):
	return np.exp(x)/(1+np.exp(x))

def compute_logistic_gradient(y, tx, w):
    """Compute gradient for the logistic regression algorithm"""
    return tx.T@(logistic_function(tx@w)-y)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def newton_method(y, tx, initial_w, batch_size, max_iters, gamma):
    "Second order Logistic Regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            H = compute_H(tx_batch, w)
            H_inv = np.linalg.inv(H)
            grad = compute_logistic_gradient(y_batch, tx_batch, w)
            w = w - gamma * H_inv@grad
    return w


#--- ASKED METHODS ---#

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad, _ = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss

def least_squares(y, tx):
    """calculate the least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    t = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    aI = lambda_ * np.identity(tx.shape[1])

    a = tx.T.dot(tx) + t
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    "First order Logistic Regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_logistic_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_):
    "First order Logistic Regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = compute_logistic_gradient(y_batch, tx_batch, w)
            grad += lambda_*w
            w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss

