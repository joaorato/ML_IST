import numpy as np
import matplotlib.pyplot as plt

def createLHS(X, P):
    
    Matrix = np.zeros((P+1, P+1))

    entry = 0
    for m in range(P+1):
        for n in range(P+1):
            for i in range(X.shape[0]):
                entry = entry + X[i]**(m+n)
            Matrix[m,n] = entry
            entry = 0

    return Matrix

def createRHS(X, Y, P):
    
    Vector = np.zeros((P+1, 1))

    entry = 0
    for m in range(P+1):
        for i in range(Y.shape[0]):
            entry = entry + Y[i]*X[i]**m
        Vector[m] = entry
        entry = 0

    return Vector

#QUESTIONS 2-3

""" x = np.load("data1_x.npy")
y = np.load("data1_y.npy")

A = createLHS(x, 1)
b = createRHS(x, y, 1)

beta = np.matmul(np.linalg.inv(A), b)

#print(np.linalg.inv(A))
#print(b)
print(beta)

points = np.linspace(-1, 1, 20)

f = beta[0] + beta[1]*points

plt.scatter(x, y)
plt.plot(points, f, color='red')
plt.show()

print('SSE')

f = beta[0] + beta[1]*x

SSE = 0
for i in range(x.shape[0]):
    SSE = SSE + (f[i] - y[i])**2

print(SSE) """

#QUESTION 4

""" x = np.load("data2_x.npy")
y = np.load("data2_y.npy")

A = createLHS(x, 2)
b = createRHS(x, y, 2)

beta = np.matmul(np.linalg.inv(A), b)

#print(np.linalg.inv(A))
#print(b)
print(beta)

points = np.linspace(-1, 1, 50)

f = beta[0] + beta[1]*points + beta[2]*points**2

plt.scatter(x, y)
plt.plot(points, f, color='red')
plt.show()

print('SSE')

f = beta[0] + beta[1]*x + beta[2]*x**2

SSE = 0
for i in range(x.shape[0]):
    SSE = SSE + (f[i] - y[i])**2

print(SSE) """

#QUESTION 5

x = np.load("data2a_x.npy")
y = np.load("data2a_y.npy")

A = createLHS(x, 2)
b = createRHS(x, y, 2)

beta = np.matmul(np.linalg.inv(A), b)

print(np.linalg.inv(A))
print(b)
print(beta)

points = np.linspace(-1, 1, 50)

f = beta[0] + beta[1]*points + beta[2]*points**2

plt.scatter(x, y)
plt.plot(points, f, color='red')
plt.show()

f = beta[0] + beta[1]*x + beta[2]*x**2

SSE = 0
squaredErr = 0
maxSquaredErr = 0
index = 0
for i in range(x.shape[0]):
    squaredErr = (f[i] - y[i])**2
    SSE = SSE + squaredErr
    if squaredErr > maxSquaredErr:
        maxSquaredErr = squaredErr

print('SSE with outliers')
print(SSE)

print('SSE without outliers')
print(SSE - maxSquaredErr)