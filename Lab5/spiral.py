import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from svm_plot import plot_contours

spiral_x = np.load('spiral_X.npy')
spiral_y = np.load('spiral_Y.npy')

### Polynomial classifier ###

""" accuracy = {}
support_vec = {}
for i in range(10):
    classifier = SVC(max_iter = 100000, kernel = 'poly', degree = float(i/2 + 1))
    classifier.fit(spiral_x, spiral_y)
    prediction = classifier.predict(spiral_x)

    accuracy['p = ' + str(i/2 + 1)] = accuracy_score(spiral_y, prediction)
    support_vec['p = ' + str(i/2 + 1)] = len(classifier.support_vectors_)
 

    print('Accuracy: ', accuracy)
    print('Support Vectors: ', support_vec) 
# Melhor sao p = 2 e p = 2.5, ambos com accuracy = 0.66

classifier_2 = SVC(max_iter = 100000, kernel = 'poly', degree = 2)
classifier_2.fit(spiral_x, spiral_y)
print(accuracy_score(spiral_y, classifier_2.predict(spiral_x)))
plot_contours(clf = classifier_2, points = spiral_x)
 """

### Gaussian RBF Classifier ###

gauss_classifier = SVC(max_iter = 100000, kernel = 'rbf', gamma = 0.01)
gauss_classifier.fit(spiral_x, spiral_y)
print(accuracy_score(spiral_y, gauss_classifier.predict(spiral_x)), '   ', len(gauss_classifier.support_vectors_))
plot_contours(clf = gauss_classifier, points = spiral_x) 