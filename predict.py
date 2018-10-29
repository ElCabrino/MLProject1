import numpy as np

from costs import *



def predict_logistic(x, w, submission=False):
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

    n = len(ws)

    ys = [predict(xs[i], ws[i]) for i in range(n)]

    ys_with_ids = np.concatenate([[ids[i], ys[i]] for i in range(n)], axis=1).T
    ys_with_ids = ys_with_ids[ys_with_ids[:, 0].argsort()]

    return ys_with_ids[:, 1]
