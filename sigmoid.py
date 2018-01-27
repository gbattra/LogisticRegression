import numpy as np

def sigmoid(z):

    # compute sigmoid value for each prediction
    g = 1 / (1 + np.exp(-z))

    return g
