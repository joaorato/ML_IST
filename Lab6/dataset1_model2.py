#SVM

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix)

x_train = np.load('dataset1_xtrain.npy')
y_train = np.load("dataset1_ytrain.npy")
x_test = np.load("dataset1_xtest.npy")
y_test = np.load("dataset1_ytest.npy")

classifier = SVC(max_iter = 500000, kernel = 'linear') #, degree = float(i + 1))
classifier.fit(x_train, y_train.ravel())
prediction = classifier.predict(x_test)
 
accuracy = accuracy_score(y_test, prediction)
support_vec = len(classifier.support_vectors_)
print('Linear')
print('Accuracy: ', accuracy)
print('Support Vectors: ', support_vec)
print(confusion_matrix(y_test, prediction))

gauss_classifier = SVC(max_iter = 200000, C = 1300, kernel = 'rbf', gamma = 0.0003) #this was found to be the best value for C and gamma
gauss_classifier.fit(x_train, y_train.ravel())
prediction = gauss_classifier.predict(x_test)

print('\nGaussian')
print('Accuracy: ', accuracy_score(y_test, prediction), '\nSupport Vectors', len(gauss_classifier.support_vectors_))
print(confusion_matrix(y_test, prediction))

#plot_contours(clf = gauss_classifier, points = x_test) doesn't work because it is 17-dimensional