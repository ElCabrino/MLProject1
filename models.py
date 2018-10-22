# This file contains all models defined using the Model class.

from features import *
from helpers import *
from costs import *
from model import *

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
