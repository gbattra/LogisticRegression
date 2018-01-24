import numpy as np
from sigmoid import sigmoid


def gradient_descent_multi(X, y, theta, alpha, iterations, L, c):
    # initialize J_history
    J_history = np.zeros(iterations)

    # get size of training data
    m = y.shape[0]

    for iter in range(0, iterations):
        # compute predictions
        z = X.dot(theta)
        h = sigmoid(z)

        # compute the cost
        J_history[iter] = (1 / m) * (((-y.T * np.log(h)) - (1 - y.T) * np.log(1 - h)).sum())

        # computer gradient
        grad = ((h - y).T.dot(X)).T

        # update theta
        theta = theta - (alpha * grad)

    return theta, J_history