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

def predict_logistic(x, w, submission=False):
    y_pred = logistic_function(x @ w)
    negative_replacement = -1 if submission else 0
    y_pred[np.where(y_pred <= 0.5)] = negative_replacement
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

def predict_values(x, w):
    y_pred = np.dot(x, w)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

def split_predict(predicts, xs, ws, ids):

    n = len(ws)

    ys = [predicts[i](xs[i], ws[i]) for i in range(n)]

    ys_with_ids = np.concatenate([np.concatenate([ys[i], ids[i]], axis=1) for i in range(n)], axis=0)
    ys_with_ids = np.sort(ys_with_ids, axis=0)

    return ys_with_ids[:, 1]

def compute_error_count(predict):

    def inner_function(y, x, w):

        y_pred = predict(x, w)
        incorrect = np.where(y_pred != y, 1, 0)
        return np.sum(incorrect) / y.shape[0]

    return inner_function

def compute_logistic_error(y, x, w):
    y_pred = logistic_function(x @ w)
    return - (y @ np.log(y_pred) + (1 - y) @ np.log(1 - y_pred)) / y.shape[0]

def sigmoid(x):

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

    return np.vectorize(sigmoid)(x)
