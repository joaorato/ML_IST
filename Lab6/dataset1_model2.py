#SVM

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

x_train = np.load('dataset1_xtrain.npy')
y_train = np.load("dataset1_ytrain.npy")
x_test = np.load("dataset1_xtest.npy")
y_test = np.load("dataset1_ytest.npy")

#accuracy = {}
#support_vec = {}
#for i in range(10):
classifier = SVC(max_iter = 100000, kernel = 'linear') #, degree = float(i + 1))
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)

#accuracy['p = ' + str(i + 1)] = accuracy_score(y_test, prediction)
#support_vec['p = ' + str(i + 1)] = len(classifier.support_vectors_)
 
accuracy = accuracy_score(y_test, prediction)
support_vec = len(classifier.support_vectors_)

print('Accuracy: ', accuracy)
print('Support Vectors: ', support_vec)

gauss_classifier = SVC(max_iter = 100000, C = 1300, kernel = 'rbf', gamma = 0.0003) #this was found to be the best value for C and gamma
gauss_classifier.fit(x_train, y_train.ravel())
print(accuracy_score(y_test, gauss_classifier.predict(x_test)), '   ', len(gauss_classifier.support_vectors_))
#FALTA MAIS UM CRITERIO DE PERFORMANCE EVALUATION

#plot_contours(clf = gauss_classifier, points = x_test) doesn't work because it is 17-dimensional