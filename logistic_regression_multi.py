# this is a logistic regression multi-class classifer
# it determines types of erythemato-squamous diseases

import numpy as np
import pandas as pd
import scipy
import datacleaner
from sigmoid import sigmoid

# import and clean data
data = pd.read_csv('dataset_multi.csv');
clean_data = datacleaner.autoclean(data, True).values
X = np.matrix(clean_data[:, 0:34])