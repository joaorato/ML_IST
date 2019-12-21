#K-NEAREST NEIGHBOURS

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE, RandomOverSampler
import scikitplot as skplt

x_train = np.load('dataset2_xtrain.npy')
y_train = np.load("dataset2_ytrain.npy")
x_test = np.load("dataset2_xtest.npy")
y_test = np.load("dataset2_ytest.npy")


# CROSS-VALIDATION - it is here that all hyperparameters are decided, even the kernel
kfold = StratifiedKFold(n_splits=5, random_state=1)

balanced_acc = []
fscore = []
conf_matrix = np.array([[0,0],[0,0]])

oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=1)
#oversampler = SMOTE(k_neighbors=5 ,random_state=1)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

kNeigh = KNeighborsClassifier(n_neighbors=15)

for train_index, val_index in kfold.split(x_train, y_train):
    X_fold_train, X_fold_val = x_train[train_index, :], x_train[val_index, :]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    
    # Oversample fold_train here (use imbalanced-learn library)
    X_fold_train_resampled, y_fold_train_resampled = oversampler.fit_resample(X_fold_train, y_fold_train)

    kNeigh.fit(X_fold_train_resampled, y_fold_train_resampled)
    prediction = kNeigh.predict(X_fold_val)

    conf_matrix += confusion_matrix(y_fold_val, prediction)
    balanced_acc.append(balanced_accuracy_score(y_fold_val, prediction))
    fscore.append(f1_score(y_fold_val, prediction))

print(conf_matrix, '\nBalanced Accuracy:', np.mean(balanced_acc), '\nF1 Score:', np.mean(fscore))



kNeigh.fit(x_train, y_train)
y_predicted = kNeigh.predict(x_test)

#print('accuracy: ', accuracy_score(y_test, y_predicted))
print('\nTest set:\n')
print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_predicted))
print('f measure = ', f1_score(y_test, y_predicted))
conf_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion matrix : \n", conf_matrix)

skplt.metrics.plot_roc(y_test, kNeigh.predict_proba(x_test), plot_micro=False, plot_macro=False, classes_to_plot=[1])
plt.show()
""" fpr, tpr, thresholds = roc_curve(y_test, y_predicted)

plt.figure(1)
plt.plot(fpr, tpr)
plt.show() """