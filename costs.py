# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_mse(y, tx, w):
    e = y - tx@w
    return 1/2*np.mean(e**2)

def compute_mae(y, tx, w):
    e = y - tx@w
    return np.mean(np.abs(e))

def compute_rmse(y, tx, w):
    return np.sqrt(2*compute_mse(y,tx,w))
