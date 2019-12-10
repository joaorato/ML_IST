#SVM

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

x_train = np.load('dataset2_xtrain.npy')
y_train = np.load("dataset2_ytrain.npy")
x_test = np.load("dataset2_xtest.npy")
y_test = np.load("dataset2_ytest.npy")

classifier = SVC(max_iter = 100000, kernel = 'linear')
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)

print("Linear")
print('Accuracy: ', accuracy_score(y_test, prediction))
print('Support Vectors: ', len(classifier.support_vectors_))
print('balanced accuracy = ', balanced_accuracy_score(y_test, prediction))
print('f measure = ', f1_score(y_test, prediction))
fpr, tpr, thresholds = roc_curve(y_test, prediction)

plt.figure(1)
plt.plot(fpr, tpr)
plt.show()


gauss_classifier = SVC(max_iter = 100000, C = np.inf, kernel = 'rbf', gamma = 0.0003) #this was found to be the best value for C and gamma
gauss_classifier.fit(x_train, y_train.ravel())
prediction = gauss_classifier.predict(x_test)

print("Gauss")
print('Accuracy: ', accuracy_score(y_test, prediction))
print('Support Vectors: ', len(gauss_classifier.support_vectors_))
print('balanced accuracy = ', balanced_accuracy_score(y_test, prediction))
print('f measure = ', f1_score(y_test, prediction))
conf_matrix = confusion_matrix(y_test, prediction)
print("Confusion matrix : \n", conf_matrix)
fpr, tpr, thresholds = roc_curve(y_test, prediction)

plt.figure(2)
plt.plot(fpr, tpr)
plt.show()

#plot_contours(clf = gauss_classifier, points = x_test) doesn't work because it is 17-dimensional