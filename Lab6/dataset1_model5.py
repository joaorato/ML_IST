#K-NEAREST NEIGHBOURS

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

x_train = np.load('dataset1_xtrain.npy')
y_train = np.load("dataset1_ytrain.npy")
x_test = np.load("dataset1_xtest.npy")
y_test = np.load("dataset1_ytest.npy")

kNeigh = KNeighborsClassifier(n_neighbors=25)

kNeigh.fit(x_train, y_train)
accuracy = kNeigh.score(x_test, y_test)

print(accuracy)