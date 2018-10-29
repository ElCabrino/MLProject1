from algebra import *
from cache import *
from costs import *
from features import *
from gradients import *
from helpers import *
from model import *
from splits import *

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

SUB_SAMPLE = False
CACHE_DIR = "test/cache/" if SUB_SAMPLE else "cache/"
SUBMISSIONS_DIR = "test/submissions/" if SUB_SAMPLE else "submissions/"

y, x, ids = load_csv_data('data/train.csv', SUB_SAMPLE)


#split the data according to the field "jet number"
y_split, x_split, _ = split_data(y, x, ids)

#best parameters we found for each of the 5 splits
h_array = [
    { 'degree': -2, 'lambda': 1e-8, 'k_fold': 4, 'seed': 0 }, #jet number = NaN
    { 'degree': 12, 'lambda': 1e-7, 'k_fold': 4, 'seed': 0 }, #jet number = 0
    { 'degree': 11, 'lambda': 1e-4, 'k_fold': 4, 'seed': 0 }, #jet number = 1
    { 'degree': 11, 'lambda': 1e-8, 'k_fold': 4, 'seed': 0 }, #jet number = 2
    { 'degree': 10, 'lambda': 1e-8, 'k_fold': 4, 'seed': 0 }  #jet number = 3
]

#files where we stored the results for each splits
caches = [Cache(CACHE_DIR + f'clean_standardize_expand_cross_validate_ridge_regression_analytical_mse_split{i}') for i in range(5)]

#compute the weights and losses for each of 5 models we use with cross validation and ridge regression
res = [
	#applies a grid search to the cartesian product of the parameters 'hs', here we only have 1 element in our grid
    evaluate(
		#clean is the function used for data preprocessing 
        clean = clean_standardize_expand,
		#fit is the function that tries to fit our data, we use analytical ridge regression and
		#cross validation. If the results are already computed for a set of parameters, fit_with_cache
		#will return the corresponding values by looking at the cache file
        fit   = fit_with_cache(cross_validate(ridge_regression_analytical, ridge_mse), caches[i]), 
        x     = x_split[i],
        y     = y_split[i], 
        hs    = h_array[i]
    )[0] for i in range(5)
]

#gets the weights
ws = [r['w'] for r in res]

#loads test dataset
y_test, x_test, ids_test = load_csv_data('data/test.csv', SUB_SAMPLE)

#split the data in the same way as the training set
_, x_test_split, ids_test_split = split_data(y_test, x_test, ids_test)

#preprocesses the test set the same way as the training set
x_test_split_prep = [clean_standardize_expand(None, x_test_split[i], h_array[i])[1] for i in range(5)]

#computes our predictions
y_pred = split_predict(predict_values, x_test_split_prep, ws, ids_test_split)

#create the submission file
create_csv_submission(ids_test, y_pred, SUBMISSIONS_DIR + 'clean_standardize_expand_cross_validate_ridge_regression_analytical_mse_split')
