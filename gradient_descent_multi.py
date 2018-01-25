import numpy as np
from sigmoid import sigmoid


def gradient_descent_multi(X, y, theta, alpha, L, iterations, c):

    # preprocess y to be binary where 1 is when y = c
    y_processed = np.zeros(y.shape[0])
    for t in range(y.shape[0]):
        y_processed[t] = 1 if y[t] == c else 0

    # initialize J_history
    J_history = np.zeros(iterations)

    # get size of training data
    m = y.shape[0]

    for i in range(0, iterations):
        # compute predictions
        z = X.dot(theta)
        h = sigmoid(z)

        # compute the cost
        J_history[i] = (1 / m) * (((-y_processed.T * np.log(h)) - (1 - y_processed.T) * np.log(1 - h)).sum())

        # computer gradient
        grad = (1 / m) * ((h - y_processed).T.dot(X)).T

        # update theta
        theta = theta - (alpha * grad)

    return theta, J_history