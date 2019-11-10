import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score
x_train = np.load("data1_xtrain.npy")
y_train = np.load("data1_ytrain.npy")

x_test = np.load("data1_xtest.npy")
y_test = np.load("data1_ytest.npy")

#UNCOMMENT FOR SCATTER PLOT
# plt.figure(1)
# plt.scatter(x_train[:50,0], x_train[:50,1], c='red', label='Class 1 train')
# plt.scatter(x_train[50:100,0], x_train[50:100,1], c='blue', label='Class 2 train')
# plt.scatter(x_train[100:150,0], x_train[100:150,1], c='green', label='Class 3 train')

# plt.scatter(x_test[:50,0], x_test[:50,1], c='orange', label='Class 1 test')
# plt.scatter(x_test[50:100,0], x_test[50:100,1], c='purple', label='Class 2 test')
# plt.scatter(x_test[100:150,0], x_test[100:150,1], c='yellow', label='Class 3 test')

# plt.xlim(-4, 7.5)
# plt.ylim(-4, 7.5)
# plt.legend()

# plt.show()

x_train_mean0_class1 = np.mean(x_train[:50,0])
x_train_var0_class1 = np.var(x_train[:50,0])

x_train_mean0_class2 = np.mean(x_train[50:100,0])
x_train_var0_class2 = np.var(x_train[50:100,0])

x_train_mean0_class3 = np.mean(x_train[100:150,0])
x_train_var0_class3 = np.var(x_train[100:150,0])

x_train_mean1_class1 = np.mean(x_train[:50,1])
x_train_var1_class1 = np.var(x_train[:50,1])

x_train_mean1_class2 = np.mean(x_train[50:100,1])
x_train_var1_class2 = np.var(x_train[50:100,1])

x_train_mean1_class3 = np.mean(x_train[100:150,1])
x_train_var1_class3 = np.var(x_train[100:150,1])

x_normal0_class1 = multivariate_normal.pdf(x_test[:,0], mean=x_train_mean0_class1, cov=x_train_var0_class1)
x_normal0_class2 = multivariate_normal.pdf(x_test[:,0], mean=x_train_mean0_class2, cov=x_train_var0_class2)
x_normal0_class3 = multivariate_normal.pdf(x_test[:,0], mean=x_train_mean0_class3, cov=x_train_var0_class3)

x_normal1_class1 = multivariate_normal.pdf(x_test[:,1], mean=x_train_mean1_class1, cov=x_train_var1_class1)
x_normal1_class2 = multivariate_normal.pdf(x_test[:,1], mean=x_train_mean1_class2, cov=x_train_var1_class2)
x_normal1_class3 = multivariate_normal.pdf(x_test[:,1], mean=x_train_mean1_class3, cov=x_train_var1_class3)

plt.plot(x_test[:,0], x_normal0_class1, 'o')
plt.plot(x_test[:,0], x_normal0_class2, 'o')
plt.plot(x_test[:,0], x_normal0_class3, 'o')
plt.plot(x_test[:,1], x_normal1_class1, 'o')
plt.plot(x_test[:,1], x_normal1_class2, 'o')
plt.plot(x_test[:,1], x_normal1_class3, 'o')
plt.show()

chosen_class = np.zeros((150))

for i in range(150):
    chosen_class[i] = 1 + np.argmax([x_normal0_class1[i]*x_normal1_class1[i], x_normal0_class2[i]*x_normal1_class2[i], x_normal0_class3[i]*x_normal1_class3[i]])

plt.plot(np.linspace(0, 150, 150, endpoint=False), chosen_class)
plt.show()

print('accuracy = ', accuracy_score(y_test, chosen_class))