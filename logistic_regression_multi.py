# this is a logistic regression multi-class classifier
# it predicts which skin condition (of 6 total conditions or classes) is represented by input X

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import datacleaner
from sigmoid import sigmoid
from train_classifier import train_classifier

# import and clean data
data = pd.read_csv('dataset_multi.csv')
clean_data = datacleaner.autoclean(data, True).values
X = np.matrix(clean_data[:, 0:34])
y = np.matrix(clean_data[:, 34:35])

# get size of training data
m = y.shape[0]

# add ones to X
X0 = np.ones((X.shape[0], 1))
X = np.hstack((X0, X))

# get number of labels and set lambda for regularization term
num_labels = y.max()
L = 1

# initialize learning params
alpha = 0.01
iterations = 5000

# get trained Theta matrix
Theta, J_histories = train_classifier(X, y, num_labels, L, iterations, alpha)

# plot J_histories to make sure gradient descent worked
for i in range(0, num_labels):
    plt.figure('J_history ' + str(i + 1))
    plt.plot(range(iterations), J_histories[i])

# run predictions and calculate accuracy
z = X.dot(Theta.T)
h = sigmoid(z)

# calculate accuracy of classifier
predicted_classes = h.argmax(axis=1) + 1  # plus one because this returns a zero indexed array but classes start with 1
err = np.abs((predicted_classes - y)).sum()
acc = (m - err) / m

# print accuracy (~ 97%)
print('Accuracy: ' + str(acc))
