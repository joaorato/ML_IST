#K-NEAREST NEIGHBOURS

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

x_train = np.load('dataset1_xtrain.npy')
y_train = np.load("dataset1_ytrain.npy")
x_test = np.load("dataset1_xtest.npy")
y_test = np.load("dataset1_ytest.npy")

kNeigh = KNeighborsClassifier(n_neighbors=25)

kNeigh.fit(x_train, y_train)
accuracy = kNeigh.score(x_test, y_test)

print(accuracy)

parameters = {'n_neighbors':[1, 5, 10, 15, 20, 25, 30, 50, 75, 100]}
kNeigh = KNeighborsClassifier()
clf = GridSearchCV(kNeigh, parameters)
clf.fit(x_train, y_train.ravel())
print(clf.cv_results_.items())