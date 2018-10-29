# This file contains everything related to computing (stochastic)
# gradient descent.
import numpy as np
from costs import *
from splits import *


def reg_logistic_regression(y, tx, initial_w, batch_size, max_iters, gamma, lambda_):
    "First order Logistic Regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad = compute_logistic_gradient(y_batch, tx_batch, w)
            grad += lambda_ * w
            w = w - gamma * grad
    return w

def logistic_regression(y, tx, initial_w, batch_size, max_iters, gamma, seed):
    "First order Logistic Regression with SGD"

    w = initial_w
    seed_iter = seed

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1, seed=seed_iter):

            grad = compute_logistic_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad

        if n_iter % 10000 == 0:
            err = compute_logistic_error(y, tx, w)
            print(f'iteration {n_iter} - err = {err}')


        seed_iter += 1

    return w

def logistic_regression_bold(y, tx, initial_w, batch_size, max_iters, gamma, seed):
    "First order Logistic Regression with SGD"
    w = initial_w

    previous_err = float('inf')
    gamma_iter = gamma
    seed_iter = seed
    prev_w = w

    delta = 1.1

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1, seed=seed_iter):

            grad = compute_logistic_gradient(y_batch, tx_batch, w)

            w = w - gamma_iter * grad

        if n_iter % 10000 == 0:

            err = compute_logistic_error(y, tx, w)

            if err > previous_err:
                w = prev_w
                gamma_iter = gamma_iter / 2
                previous_err = err
            else:
                gamma_iter = gamma_iter * delta
                previous_err = err

            prev_w = w

            print(f'iteration {n_iter} - err = {err}, gamma = {gamma_iter}')

        seed_iter = seed_iter + 1

    return w

#Stochastic Gradient Descent with Lasso regularization
def lasso_stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, lambd):
    """Stochastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            #Lasso regularization
            grad += [lambd * np.sign(w_i) if w_i != 0 else 0 for w_i in w]
            w = w - gamma * grad
    return w

#Stochastic Gradient Descent
def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
    return w

#Stochastic Gradient Descent
def stochastic_gradient_descent_precision(y, tx, initial_w, batch_size, precision, gamma, loss_f):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    # ws = [initial_w]
    # losses = []
    w = initial_w
    loss = float("inf")
    diff = float("inf")

    while diff > precision:
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad

            diff = loss
            # calculate loss
            loss = loss_f(y, tx, w)
            # update diff
            diff = diff - loss

    return w, loss
