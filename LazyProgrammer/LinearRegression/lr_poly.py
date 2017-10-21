import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:,1], Y)
plt.show()

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

v1 = Y - Yhat
v2 = Y - Y.mean()
R2 = 1 - v1.dot(v1)/v2.dot(v2)

print("r- square " + str(R2))

plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:,1]), sorted(Yhat))
plt.show()
