import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

Yhat = a * X + b

v1 = Y - Yhat
v2 = Y - Y.mean()

R2 = 1 - v1.dot(v1)/v2.dot(v2)

print("R2 " + str(R2))

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()
