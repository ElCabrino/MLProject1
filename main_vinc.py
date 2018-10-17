import numpy as np
import matplotlib.pyplot as plt

# Import data

import datetime
from helpers import *
from costs import *
from implementations import *

#Load data for training
y, x, ids = load_csv_data("train.csv", sub_sample=False)
x, _, _ = standardize(x)
tx = np.c_[np.ones(x.shape[0]), x]

#Load data for testing
y_test, x_test, ids_test = load_csv_data("test.csv", sub_sample=False)
x_test, _, _ = standardize(x_test)
tx_test = np.c_[np.ones(x_test.shape[0]), x_test]

print(tx.shape)
print()
print(len(y))

# Define the parameters of the algorithm.
max_iters = 100000
gamma = 0.08

# Initialization
#w_initial = np.zeros(tx.shape[1])
w_initial = np.random.rand(tx.shape[1])
# Start gradient descent.
start_time = datetime.datetime.now()
gradient_losses, gradient_ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

#Making predictions on the testing set
y_pred = predict_labels(gradient_ws[-1], tx_test)

create_csv_submission(ids_test, y_pred, "GD_first_try.csv")