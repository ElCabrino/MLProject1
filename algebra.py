# This file contains all functions related to solving the normal
# function analytically using linear algebra

def least_squares(y, tx):
    """calculate the least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""

    t = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    aI = lambda_ * np.identity(tx.shape[1])

    a = tx.T.dot(tx) + t
    b = tx.T.dot(y)

    return np.linalg.solve(a, b)
