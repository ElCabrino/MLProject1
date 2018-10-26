# -*- coding: utf-8 -*-
import numpy as np

def compute_mse(y, tx, w):
    e = y - tx@w
    return 1/2*np.mean(e**2)

def compute_mae(y, tx, w):
    e = y - tx@w
    return np.mean(np.abs(e))

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y,tx,w))

def compute_error_count(y, x, w):
    return np.sum(np.vectorize(lambda xw, y: 0 if xw * y > 0 else 1)(x @ w, y))

def compute_logistic_error(y, x, w):
    y_pred = logistic_function(x @ w)
    return - (y @ np.log(y_pred) + (1 - y) @ np.log(1 - y_pred))

def sigmoid(x):

    threshold = 0.00001

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

    return np.vectorize(sigmoid)(x)
