import numpy as np
import matplotlib.pyplot as plt
from gradient_descent_multi import gradient_descent_multi


def train_classifier(X, y, num_labels, L, iterations, alpha):

    # get m and n for Theta size
    m = X.shape[0]
    n = X.shape[1]

    # Theta is the matrix where we will store the values of each theta we train
    Theta = np.zeros((num_labels, n))

    # train Theta for each class
    for c in range(0, num_labels):

        # initialize theta
        initial_theta = np.zeros((n, 1))

        # train theta for this class
        theta, J_history = gradient_descent_multi(X, y, initial_theta, alpha, L, iterations, c + 1)

        # add this theta to Theta matrix
        Theta[c, :] = np.array(theta.T[0, :])

        # plot J_history to make sure gradient descent multi is working properly
        plt.figure('J_history ' + str(c + 1))
        plt.plot(range(iterations), J_history)
        plt.show()

    return Theta
