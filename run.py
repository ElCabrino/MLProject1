from cache import *
from costs import *
from features import *
from helpers import *
from evaluate import *
from validate import *
from implementations import *

import csv
import numpy as np

SUB_SAMPLE = False
CACHE_DIR = "test/cache/" if SUB_SAMPLE else "cache/"
SUBMISSIONS_DIR = "test/submissions/" if SUB_SAMPLE else "submissions/"

y, x, ids = load_csv_data('data/train.csv', SUB_SAMPLE)


# Split the data according to the mass of the Higgs Boson candidate and the
# field "jet number"
y_split, x_split, _ = split_data(y, x, ids)

# List of best parameters we found for each of the 5 splits
h_array = [
    { 'degree': -2, 'lambda': 1e-8, 'k_fold': 4, 'seed': 0 }, #jet number = NaN
    { 'degree': 12, 'lambda': 1e-7, 'k_fold': 4, 'seed': 0 }, #jet number = 0
    { 'degree': 11, 'lambda': 1e-4, 'k_fold': 4, 'seed': 0 }, #jet number = 1
    { 'degree': 11, 'lambda': 1e-8, 'k_fold': 4, 'seed': 0 }, #jet number = 2
    { 'degree': 10, 'lambda': 1e-8, 'k_fold': 4, 'seed': 0 }  #jet number = 3
]

# Files where we stored the results for each splits. This avoids re-computing
# the whole pipeline every time. If you want to start from scratch,
caches = [Cache(CACHE_DIR + f'clean_standardize_expand_cross_validate_ridge_regression_analytical_mse_split{i}') for i in range(5)]

# Compute the weights and losses for each of 5 models we use with cross validation and ridge regression
res = [
    # Our 'evaluate' function runs a full pipeline (clean, fit, validate) on
    # all the hyperparameters passed, and was therefore used for fine-tuning
    # In this case, we only have one value for each parameter, so it will return
    # the results for these parameters only.
    evaluate(
		#fit is the function that tries to fit our data, we use analytical ridge regression and
		#cross validation. If the results are already computed for a set of parameters, fit_with_cache
		#will return the corresponding values by looking at the cache file
        clean_and_fit_with_cache(
            # We clean each dataset with our function that removes the errors,
            # removes the outliers, standardizes the features and expand the
            # polynomials.
            clean_standardize_expand,
            # We fit our models using Ridge Regression (calculated analytically)
            # and cross-validating 4 times with the mean-squared error.
            cross_validate(ridge_regression_weights, mse_and_ridge),
            caches[i]),
        y  = y_split[i],
        x  = x_split[i],
        hs = h_array[i]
    )[0] for i in range(5)
]

# Extract the weights from the result
ws = [r['w'] for r in res]

# Load test data, split it using the same categories and clean those
# categories with the same cleaning function as before.
y_test, x_test, ids_test = load_csv_data('data/test.csv', SUB_SAMPLE)
_, x_test_split, ids_test_split = split_data(y_test, x_test, ids_test)
x_test_split_prep = [clean_standardize_expand(None, x_test_split[i], h_array[i])[1] for i in range(5)]

# Calculates predictions
y_pred = split_predict(predict_values, x_test_split_prep, ws, ids_test_split)

# Create the submission file
create_csv_submission(ids_test, y_pred, SUBMISSIONS_DIR + 'clean_standardize_expand_cross_validate_ridge_regression_analytical_mse_split')
