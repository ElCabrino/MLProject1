# This file contains all models defined using the Model class.
import numpy as np
from math import isnan, inf
from features import *
from helpers import *
from costs import *
from model import *
from algebra import *
from gradients import *

class LeastSquare_MSE_Model(Model):

    def prepare(self, x, y):

        return clean_data(self.raw_x), y

    def fit(self, x, y, h={}):

        w = least_squares(self.y, self.x)
        mse = compute_mse(self.y, self.x, w)

        return {
            "mse": mse
        }, w


class LeastSquare_MSE_Degree_Model(Model):

    def prepare(self, x, y):

        return clean_data(self.raw_x), y

    def fit(self, x, y, h={}):

        degree = int(h['degree'])

        tx = build_poly(x, degree)

        w = least_squares(y, tx)
        mse = compute_mse(y, tx, w)

        return {
            "mse": mse
        }, w

class RidgeRegression_MSE_Degree_Model(Model):

    def prepare(self, x, y):

        return clean_data(self.raw_x), y

    def fit(self, x, y, h={}):

        degree = int(h['degree'])
        lambda_ = float(h['lambda'])

        tx = build_poly(x, degree)

        w = ridge_regression(y, tx, lambda_)
        mse = compute_mse(y, tx, w)

        return {
            "mse": mse
        }, w

class StochasticGradientDescent_MSE_Degree_Model(Model):

    def prepare(self, x, y):

        return clean_data(self.raw_x), y

    def fit(self, x, y, h={}):
        
        batch_size = int(h['batch_size'])
        n_iters = int(h['n_iters'])
        degree = int(h['degree'])
        gamma = float(h['gamma'])

        tx = build_poly(x, degree)
        
        initial_w = np.zeros(tx.shape[1])

        w = stochastic_gradient_descent(y, tx, initial_w, batch_size, n_iters, gamma)
        mse = compute_mse(y, tx, w)

        if isnan(mse):
            mse = inf
        
        return {
            "mse": mse
        }, w

class Lasso_SGD_MSE_Degree_Model(Model):

    def prepare(self, x, y):

        return clean_data(self.raw_x), y

    def fit(self, x, y, h={}):
        
        batch_size = int(h['batch_size'])
        n_iters = int(h['n_iters'])
        degree = int(h['degree'])
        lambda_ = float(h['lambda'])
        gamma = float(h['gamma'])

        tx = build_poly(x, degree) 

        initial_w = np.zeros(tx.shape[1])

        w = lasso_stochastic_gradient_descent(y, tx, initial_w, batch_size, n_iters, gamma, lambda_)
        mse = compute_mse(y, tx, w)

        if isnan(mse):
            mse = inf
        
        return {
            "mse": mse
        }, w
        
class First_Order_Logistic_Regression_Model(Model):

    def prepare(self, x, y):

        return clean_data(self.raw_x), y

    def fit(self, x, y, h={}):
        
        batch_size = int(h['batch_size'])
        n_iters = int(h['n_iters'])
        degree = int(h['degree'])
        gamma = float(h['gamma'])

        tx = build_poly(x, degree)
        
        initial_w = np.zeros(tx.shape[1])

        w = logistic_regression(y, tx, initial_w, batch_size, n_iters, gamma)
        mse = compute_mse(y, tx, w)

        if isnan(mse):
            mse = inf
        
        return {
            "mse": mse
        }, w
