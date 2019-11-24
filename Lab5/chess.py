import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from svm_plot import plot_contours

chess_x = np.load('chess33_X.npy')
chess_y = np.load('chess33_Y.npy')

classifier = SVC(max_iter = 100000, C = np.inf, kernel = 'rbf', gamma = 0.01)
classifier.fit(chess_x, chess_y)
print(accuracy_score(chess_y, classifier.predict(chess_x)), len(classifier.support_vectors_)) # accuracy at 1 and 10 support vectors for gamma = 0.01
plot_contours(clf = classifier, points = chess_x) 

# print(classifier.predict(chess_x) == chess_y)