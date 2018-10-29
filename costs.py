# -*- coding: utf-8 -*-
import numpy as np


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


#   Aggregated loss functions
#   -------------------------
#
#   In a lot of cases in our exploration, we wanted to be able to analyze more
#   than one error at a time. In order to do this, we used these aggregated
#   functions that call several loss functions in parallel

def mse(y, x, w, h):

    return {
        'mse' : compute_mse(y, x, w),
        'n_err': compute_error_count(predict_values)(y, x, w)
    }


def mse_and_ridge(y, x, w, h):

    lambda_ = float(h['lambda'])

    mse = compute_mse(y, x, w)
    ridge_norm = np.linalg.norm(w, 2) * lambda_

    return {
        'mse': mse,
        'ridge_norm': ridge_norm,
        'total_loss': mse + ridge_norm,
        'n_err': compute_error_count(predict_values)(y, x, w)
    }


def mse_and_lasso(y, x, w, h):

    lambda_ = float(h['lambda'])

    mse = compute_mse(y, x, w)
    lasso_norm = np.linalg.norm(w, 1) * lambda_

    return {
        'mse': mse,
        'lasso_norm': lasso_norm,
        'total_loss': mse + lasso_norm,
        'n_err': compute_error_count(predict_values)(y, x, w)
    }


def logistic_error(y, x, w, h):

    return {
        'logistic_err': compute_logistic_error(y, x, w),
        'n_err': compute_error_count(predict_logistic)(y, x, w)
    }


def logistic_error_and_ridge(y, x, w, h):

    lambda_ = h['lambda']

    ridge_norm = np.linalg.norm(w, 2) * lambda_
    logistic_err = compute_logistic_error(y, x, w)
    n_err = compute_error_count(predict_logistic)(y, x, w)

    return {
        'logistic_err': logistic_err,
        'ridge_norm': ridge_norm,
        'total_loss': logistic_err + ridge_norm,
        'n_err': n_err
    }


def logistic_error_and_lasso(y, x, w, h):

    lambda_ = h['lambda']

    lasso_norm = np.linalg.norm(w, 1) * lambda_
    logistic_err = compute_logistic_error(y, x, w)
    n_err = compute_error_count(predict_logistic)(y, x, w)

    return {
        'logistic_err': logistic_err,
        'lasso_norm': lasso_norm,
        'total_loss': logistic_err + lasso_norm,
        'n_err': n_err
    }

#   Prediction Functions
#   --------------------
#
#   These functions are used to predict new values based on data and weights

def predict_logistic(x, w, submission=True):
    """Predict values with weights received from logistic regression"""
    y_pred = logistic_function(x @ w)
    negative_replacement = -1 if submission else 0
    y_pred[np.where(y_pred <= 0.5)] = negative_replacement
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred


def predict_values(x, w):
    """Predict values with weights received from least square"""
    y_pred = np.dot(x, w)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def split_predict(predict, xs, ws, ids):

    n = 5

    ys = [predict(xs[i], ws[i]) for i in range(n)]

    ys_with_ids = np.concatenate([[ids[i], ys[i]] for i in range(n)], axis=1).T
    ys_with_ids = ys_with_ids[ys_with_ids[:, 0].argsort()]

    return ys_with_ids[:, 1]

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
