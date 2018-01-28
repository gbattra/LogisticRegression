import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent


def train_classifier(X, y, num_labels, L, iterations, alpha):

    # get n for Theta size
    n = X.shape[1]

    # Theta is the matrix where we will store the theta values of each classifier we train
    Theta = np.zeros((num_labels, n))

    # J_histories is the matrix of the J_history for each classifier we train
    J_histories = np.zeros((num_labels, iterations))

    # train Theta for each class
    for i in range(0, num_labels):
        # get class
        c = i + 1

        # preprocess y to be binary where 1 is when y = c
        y_processed = y.copy()
        for t in range(y.shape[0]):
            y_processed[t] = 1 if y[t] == c else 0

        # initialize theta
        initial_theta = np.zeros((n, 1))

        # train theta for this class
        theta, J_history = gradient_descent(X, y_processed, initial_theta, alpha, iterations)

        # add this theta to Theta matrix
        Theta[i, :] = np.array(theta.T[0, :])

        # add this J_history to the J_histories matrix
        J_histories[i, :] = J_history

    return Theta, J_histories
