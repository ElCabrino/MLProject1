# This file contains everything related to computing (stochastic)
# gradient descent.
import numpy as np
from costs import *
from splits import *


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_logistic_gradient(y, tx, w):
    """Compute gradient for the logistic regression algorithm"""
    return tx.T @ (logistic_function(tx @ w) - y) / y.shape[0]

def compute_S(tx, w):
    """"Compute S matrix for second order logistic regression"""
    n = tx.shape[0]
    S = np.zeros([n, n])
    for i in range(n):
        sigma_xW = logistic_function(tx[i].T@w)
        S[i, i] = sigma_xW*(1-sigma_xW)
    return S

def compute_H(tx, w):
    """Compute H matrix for second order logistic regression"""
    S = compute_S(tx, w)
    return tx.T@S@tx

def newton_method(y, tx, initial_w, batch_size, max_iters, gamma):
    "Second order Logistic Regression with SGD"
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            H = compute_H(tx_batch, w)
            H_inv = np.linalg.inv(H)
            grad = compute_logistic_gradient(y_batch, tx_batch, w)
            w = w - gamma * H_inv @ grad
    return w

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
