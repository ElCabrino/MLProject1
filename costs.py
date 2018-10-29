# -*- coding: utf-8 -*-
import numpy as np

from predict import *


#   Loss functions
#   --------------
#   Below are all the different loss functions that we use in our models.
#   Note that they return a dictionary, the reason for this is that we will use
#   this dictionary in the higher-level wrapper functions like fit_with_cache,
#   and we want to associate a name with the value returned.

def compute_mse(y, tx, w):
    """compute the mean squared error for the given weights and data."""
    e = y - tx@w
    return 1 / 2 * np.mean(e ** 2)


def compute_mae(y, tx, w):
    """compute the mean absolute error for the given weights and data."""
    e = y - tx@w
    return np.mean(np.abs(e))


def compute_rmse(y, tx, w):
    """compute the mean squared error for the given weights and data."""
    return np.sqrt(2 * compute_mse(y,tx,w))


def compute_logistic_error(y, x, w):
    """compute the logistic error for the given weights and data."""

    y_pred = logistic_function(x @ w)
    return -(y @ np.log(y_pred) + (1 - y) @ np.log(1 - y_pred)) / y.shape[0]

def compute_error_count(predict):
    """
    Given a prediction function (that will map x @ w into the [-1, 1] space),
    returns a function that will compute the percentage of incorrect predictions.
    """

    def inner_function(y, x, w):

        y_pred = predict(x, w)
        incorrect = np.where(y_pred != y, 1, 0)
        return np.sum(incorrect) / y.shape[0]

    return inner_function


#   Helpers
#   -------


def sigmoid(x):
    """
    Sigmoid function for one value.

    Note: We use a threshold of 1e-10 to avoid breaking the system when values
    become too close to 0 or 1. Any value will be clamped to not approach
    0 or 1 closer than the threshold.
    """

    threshold = 1e-10

    if x > 0:
        res = 1 / (np.exp(-x) + 1)
    else:
        res = np.exp(x) / (1 + np.exp(x))

    if res < threshold:
        return threshold
    elif res > 1 - threshold:
        return 1 - threshold
    else:
        return res


def logistic_function(x):
    """Sigmoid function for a numpy.ndarray."""

    return np.vectorize(sigmoid)(x)
