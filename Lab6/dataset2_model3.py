#DECISION TREE

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

x_train = np.load('dataset2_xtrain.npy')
y_train = np.load("dataset2_ytrain.npy")
x_test = np.load("dataset2_xtest.npy")
y_test = np.load("dataset2_ytest.npy")

train_instances = x_train.shape[0]
test_instances = x_test.shape[0]
features = x_train.shape[1]
class_dimensions = y_train.shape[1]

decisionTree = DecisionTreeClassifier(criterion='entropy')
decisionTree.fit(x_train, y_train.ravel())
y_predicted = decisionTree.predict(x_test)

print('accuracy: ', decisionTree.score(x_test, y_test.ravel()))
print('Balanced accuracy: ', balanced_accuracy_score(y_test, y_predicted))
print('f measure = ', f1_score(y_test, y_predicted))
conf_matrix = confusion_matrix(y_test, y_predicted)
print("Confusion matrix : \n", conf_matrix)
fpr, tpr, thresholds = roc_curve(y_test, y_predicted)

plt.figure(1)
plt.plot(fpr, tpr)
plt.show()

# meanValue = np.mean(x_train)

# # We define two "classes" of features, if x_i < meanValue/4 it belongs to 0, it belongs to 1 for the rest of the cases (this is done so that the set length is similar)

# test_table = np.zeros((train_instances, features + 1))

# test_table[:, features] = y_train.ravel()
# counter = 0
# for i in range(train_instances):
#     for j in range(features):
#         if x_train[i, j] >= meanValue/4:
#             counter += 1
#             test_table[i, j] = 1

# feature_y_tables = np.zeros((features, 2, 2)) # there will be "feature" number of 2 x 2 tables since there are 2 feature classes and 2 classification classes

# for i in features:
#     for j in range(2):
#         for k in range(2):
#             feature_y_tables[i, j, k] = intance_count