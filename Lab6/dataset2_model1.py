#BAYES

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_curve, confusion_matrix)

x_train = np.load("dataset2_xtrain.npy")
y_train = np.load("dataset2_ytrain.npy")
x_test = np.load("dataset2_xtest.npy")
y_test = np.load("dataset2_ytest.npy")

#print(x_train.shape, y_train.shape) #(921, 17) -> (921, 1)
#print(x_test.shape, y_test.shape) #(230, 17) -> (230, 1)

train_instances = x_train.shape[0]
test_instances = x_test.shape[0]
features = x_train.shape[1]
class_dimensions = y_train.shape[1]


#THIS NEXT BLOCK COUNTS HOW MANY PATTERNS THERE ARE IN EACH CLASS

zeroCounter = 0
oneCounter = 0

for i in range(train_instances):
    if y_train[i] == 0:
        zeroCounter += 1
    elif y_train[i] == 1:
        oneCounter += 1

# print("zeros: ", zeroCounter, "\nones: ", oneCounter) #train: 432 zeros and 489 ones; test: 108 zeros and 122 ones

# since the set is fairly balanced, a BAYES CLASSIFIER can be tried out. The next block separates the data by class

x_train_class0 = np.zeros((zeroCounter, features))
y_train_class0 = np.zeros((zeroCounter, class_dimensions))

x_train_class1 = np.zeros((oneCounter, features))
y_train_class1 = np.ones((oneCounter, class_dimensions))

j = 0
k = 0

for i in range(train_instances):
    if y_train[i] == 0:
        x_train_class0[j] = x_train[i]
        j += 1
    elif y_train[i] == 1:
        x_train_class1[k] = x_train[i]
        k += 1

#Calculate mean and variance of the distributions for each feature (and class)

x_train_class0_mean = np.zeros(features)
x_train_class0_var = np.zeros(features)

x_train_class1_mean = np.zeros(features)
x_train_class1_var = np.zeros(features)

for i in range(features):
    x_train_class0_mean[i] = np.mean(x_train_class0[:, i])
    x_train_class0_var[i] = np.var(x_train_class0[:, i])
    x_train_class1_mean[i] = np.mean(x_train_class1[:, i])
    x_train_class1_var[i] = np.var(x_train_class1[:, i])

#Create the normal distributions from the data above with the test data as input

x_normal_class0 = np.zeros((features, test_instances))
x_normal_class1 = np.zeros((features, test_instances))

for i in range(features):
    x_normal_class0[i] = multivariate_normal.pdf(x_test[:,i], mean=x_train_class0_mean[i], cov=x_train_class0_var[i])
    x_normal_class1[i] = multivariate_normal.pdf(x_test[:,i], mean=x_train_class1_mean[i], cov=x_train_class1_var[i])

plt.figure(1)
plt.plot(x_test[:,9], x_normal_class0[9], 'o')
#plt.plot(x_test[:,0], x_normal_class1[0], 'o')
plt.show()

#Calculate Bayes probabilities

product_class0 = np.ones(test_instances)
product_class1 = np.ones(test_instances)

for i in range(test_instances):
    for j in range(features):
        product_class0[i] *= x_normal_class0[j][i]
        product_class1[i] *= x_normal_class1[j][i]
        
chosen_class = np.zeros(test_instances)

#calculate prior probabilities
prior_class0 = zeroCounter/train_instances
prior_class1 = oneCounter/train_instances

for i in range(test_instances):
    chosen_class[i] = np.argmax([product_class0[i]*prior_class0, product_class1[i]*prior_class1])

plt.figure(2)
plt.plot(np.linspace(0, test_instances, test_instances, endpoint=False), chosen_class)
plt.show()

print('accuracy = ', accuracy_score(y_test, chosen_class))
print('confusion matrix:\n', confusion_matrix(y_test, chosen_class))

print('balanced accuracy = ', balanced_accuracy_score(y_test, chosen_class))
print('f measure = ', f1_score(y_test, chosen_class))
fpr, tpr, thresholds = roc_curve(y_test, chosen_class)

plt.figure(3)
plt.plot(fpr, tpr)
plt.show()