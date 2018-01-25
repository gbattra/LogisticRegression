# this is a logistic regression multi-class classifer
# it determines types of erythemato-squamous diseases

import numpy as np
import pandas as pd
import scipy
import datacleaner
from sigmoid import sigmoid
from train_classifier import train_classifier

# import and clean data
data = pd.read_csv('dataset_multi.csv')
clean_data = datacleaner.autoclean(data, True).values
X = np.matrix(clean_data[:, 0:34])
y = np.matrix(clean_data[:, 34:35])

# add ones to X
X0 = np.ones((X.shape[0], 1))
X = np.hstack((X0, X))

# train model to accurately predict class
num_labels = y.max()
L = 1

# initialize learning params
alpha = 0.01
iterations = 100

# get trained Theta matrix
Theta = train_classifier(X, y, num_labels, L, iterations, alpha)


