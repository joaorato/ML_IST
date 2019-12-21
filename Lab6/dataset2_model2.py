#SVM

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

x_train = np.load('dataset2_xtrain.npy')
y_train = np.load("dataset2_ytrain.npy")
x_test = np.load("dataset2_xtest.npy")
y_test = np.load("dataset2_ytest.npy")

lin_classifier = SVC(max_iter = 100000, kernel = 'linear')

# CROSS-VALIDATION - it is here that all hyperparameters are decided, even the kernel
kfold = StratifiedKFold(n_splits=5, random_state=1)

lin_balanced_acc = []
lin_fscore = []
lin_conf_matrix = np.array([[0,0],[0,0]])


y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
for train_index, val_index in kfold.split(x_train, y_train):
    X_fold_train, X_fold_val = x_train[train_index, :], x_train[val_index, :]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    # Oversample fold_train here (use imbalanced-learn library)

    lin_classifier.fit(X_fold_train, y_fold_train)
    prediction = lin_classifier.predict(X_fold_val)

    lin_conf_matrix += confusion_matrix(y_fold_val, prediction)
    lin_balanced_acc.append(balanced_accuracy_score(y_fold_val, prediction))
    lin_fscore.append(f1_score(y_fold_val, prediction))

print(lin_conf_matrix, '\nBalanced Accuracies:', lin_balanced_acc, '\nF1 Scores:', lin_fscore)


# Now evaluate on test set
lin_classifier.fit(x_train, y_train)
prediction = lin_classifier.predict(x_test)


print("\nTest Set:\n")
#print('Accuracy: ', accuracy_score(y_test, prediction))
print('Support Vectors: ', len(lin_classifier.support_vectors_))
print('balanced accuracy = ', balanced_accuracy_score(y_test, prediction))
print('f measure = ', f1_score(y_test, prediction))
conf_matrix = confusion_matrix(y_test, prediction)
print("Confusion matrix : \n", conf_matrix)

# Use scikit-plot library to plot the actual roc_curve
""" fpr, tpr, thresholds = roc_curve(y_test, prediction)

plt.figure(1)
plt.plot(fpr, tpr)
plt.show() """

"""
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
 """
#plot_contours(clf = gauss_classifier, points = x_test) doesn't work because it is 17-dimensional