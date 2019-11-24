import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from svm_plot import plot_contours

chessN_x = np.load('chess33n_X.npy')
chessN_y = np.load('chess33n_Y.npy')

# classifier = SVC(max_iter = 100000, C = np.inf, kernel = 'rbf', gamma = 0.01)
classifier = SVC(max_iter = 100000, C = 1000, kernel = 'rbf', gamma = 0.01)
classifier.fit(chessN_x, chessN_y)
print(accuracy_score(chessN_y, classifier.predict(chessN_x)), len(classifier.support_vectors_)) # accuracy at 1 and 16 support vectors for gamma = 0.01
plot_contours(clf = classifier, points = chessN_x) 

print(classifier.predict(chessN_x) == chessN_y)