import numpy as np
import matplotlib.pyplot as plt
from gradient_descent import gradient_descent


def train_classifier(X, y, classes, L, iterations, alpha):

    # get m and n for Theta size
    m = X.shape[0]
    n = X.shape[1]

    # Theta is the matrix where we will store the values of each theta we train
    Theta = np.zeros((classes.size, n))

    # train Theta for each class
    for i in range(0, classes.size):
        # get class
        c = classes[i]

        # preprocess y to be binary where 1 is when y = c
        y_processed = y
        for t in range(y.shape[0]):
            y_processed[t] = 1 if y[t] == c else 0

        # initialize theta
        initial_theta = np.zeros((n, 1))

        # train theta for this class
        theta, J_history = gradient_descent(X, y_processed, initial_theta, alpha, iterations)

        # add this theta to Theta matrix
        Theta[i, :] = np.array(theta.T[0, :])

        # plot J_history to make sure gradient descent multi is working properly
        plt.figure('J_history ' + str(c))
        plt.plot(range(iterations), J_history)
        plt.show()

    return Theta
