#K-NEAREST NEIGHBOURS

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

x_train = np.load('dataset2_xtrain.npy')
y_train = np.load("dataset2_ytrain.npy")
x_test = np.load("dataset2_xtest.npy")
y_test = np.load("dataset2_ytest.npy")

kNeigh = KNeighborsClassifier(n_neighbors=25)

kNeigh.fit(x_train, y_train)
y_predicted = kNeigh.predict(x_test)

print('accuracy: ', accuracy_score(y_test, y_predicted))
print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_predicted))
print('f measure = ', f1_score(y_test, y_predicted))
conf_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion matrix : \n", conf_matrix)
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)

plt.figure(1)
plt.plot(fpr, tpr)
plt.show()