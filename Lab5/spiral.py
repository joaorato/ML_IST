import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from svm_plot import plot_contours

spiral_x = np.load('spiral_X.npy')
spiral_y = np.load('spiral_Y.npy')

### Polynomial classifier ###
""" 
accuracy = {}
for i in range(10):
    classifier = SVC(max_iter = 100000, kernel = 'poly', gamma = float(i/2 + 1))
    classifier.fit(spiral_x, spiral_y)
    prediction = classifier.predict(spiral_x)

    accuracy['p = ' + str(i/2 + 1)] = accuracy_score(spiral_y, prediction)
 """

# print(accuracy)  # Melhor sao p = 1.5 e p = 3, ambos com accuracy = 0.57

classifier_3 = SVC(max_iter = 100000, kernel = 'poly', gamma = 3)
classifier_3.fit(spiral_x, spiral_y)
#print(accuracy_score(spiral_y, classifier_3.predict(spiral_x)))
#plot_contours(clf = classifier_3, points = spiral_x)


### Gaussian RBF Classifier ###

gauss_classifier = SVC(max_iter = 100000, kernel = 'rbf', gamma = 1)
gauss_classifier.fit(spiral_x, spiral_y)
print(accuracy_score(spiral_y, gauss_classifier.predict(spiral_x)))
plot_contours(clf = gauss_classifier, points = spiral_x)