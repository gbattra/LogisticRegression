# this is a logistic regression multi-class classifer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import datacleaner
from sigmoid import sigmoid
from train_classifier import train_classifier

# import and clean data
data = pd.read_csv('dataset.csv')
clean_data = datacleaner.autoclean(data, True).values
X = np.matrix(clean_data[:, 0:9])
y = np.matrix(clean_data[:, 9:10])

# add ones to X
X0 = np.ones((X.shape[0], 1))
X = np.hstack((X0, X))

# get classes and set lambda for regularization term
classes = np.unique(np.asarray(y))
L = 1

# initialize learning params
alpha = 0.01
iterations = 10000

# get trained Theta matrix
Theta = train_classifier(X, y, classes, L, iterations, alpha)
