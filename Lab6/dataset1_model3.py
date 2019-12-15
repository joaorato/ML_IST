#DECISION TREE

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

x_train = np.load('dataset1_xtrain.npy')
y_train = np.load("dataset1_ytrain.npy")
x_test = np.load("dataset1_xtest.npy")
y_test = np.load("dataset1_ytest.npy")

train_instances = x_train.shape[0]
test_instances = x_test.shape[0]
features = x_train.shape[1]
class_dimensions = y_train.shape[1]

decisionTree = DecisionTreeClassifier(criterion='entropy')
decisionTree.fit(x_train, y_train.ravel())
y_predicted = decisionTree.predict(x_test)

print('accuracy: ', decisionTree.score(x_test, y_test.ravel()))
print('confusion matrix\n', confusion_matrix(y_test, y_predicted))
