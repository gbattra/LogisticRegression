import numpy as np
import matplotlib.pyplot as plt
from gradient_descent_multi import gradient_descent_multi


def train_classifier(X, y, num_labels, lamba, iterations, alpha):

    # get m and n for Theta size
    m = X.shape[0]
    n = X.shape[1]

    # Theta is the matrix where we will store the values of each theta we train
    Theta = np.zeros((num_labels, n + 1))

    # train Theta for each class
    for c in range(0, num_labels):
        # initialize theta
        initial_theta = np.zeros((n + 1, 1))

        # train theta for this class
        theta, J_history = gradient_descent_multi(X, y, initial_theta, alpha, iterations, lamba, c)

        # add this theta to Theta matrix
        Theta[c, :] = theta

        # plot J_history to make sure gradient descent multi is working properly
        plt.figure('J_history')
        plt.plot(range(iterations), J_history)

    plt.show()

    return Theta
