import numpy as np
import matplotlib.pyplot as plt

# Import data

import datetime
from helpers import *
from costs import *
from implementations import *

#Load data
y, x, ids = load_csv_data("train.csv", sub_sample=True)
tx = np.c_[np.ones(x.shape[0]), x]

print(tx.shape)
print()
print(len(y))

# Define the parameters of the algorithm.
max_iters = 100
gamma = 0.000001

# Initialization
w_initial = np.zeros(tx.shape[1])
# Start gradient descent.
start_time = datetime.datetime.now()
gradient_losses, gradient_ws = gradient_descent(y, tx, w_initial, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))