import numpy as np
from sigmoid import sigmoid


def compute_cost(X, y, theta):
    m = y.shape[0]

    # make predictions
    z = X.dot(theta)
    h = sigmoid(z)

    # initialize J and grad
    J = 0
    grad = 0

    # compute cost
    for i in range(0, m):
        J += (-y.item(i) * np.log(h.item(i))) - (1 - y.item(i)) * np.log(1 - h.item(i))
        grad += (h.item(i) - y.item(i)) * X[i, :].T

    J = (1 / m) * J
    grad = (1 / m) * grad

    return J, grad
