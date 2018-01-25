import numpy as np
from sigmoid import sigmoid


def gradient_descent(X, y, theta, alpha, iterations):
    # initialize J_history
    J_history = np.zeros(iterations)

    # get size of training data
    m = y.shape[0]

    # run gradient descent
    for i in range(0, iterations):
        # compute prediction
        z = X.dot(theta)
        h = sigmoid(z)

        # compute cost
        J_history[i] = (1 / m) * (((-y.T * np.log(h)) - (1 - y.T) * np.log(1 - h)).sum())

        # compute gradient
        grad = (1 / m) * ((h - y).T.dot(X)).T

        # update thetas
        theta = theta - (alpha * grad)

    return theta, J_history
