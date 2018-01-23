import numpy as np

def sigmoid(z):
    # placeholder matrix for storing computed values
    g = np.zeros(z.shape)

    # compute sigmoid value for each prediction
    for i in range(0, g.shape[0]):
        for v in range(0, z.shape[1]):
            g[i, v] = 1 / (1 + np.power((np.e, -z.item(i))))

    return g
