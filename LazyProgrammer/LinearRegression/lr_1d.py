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

RR = Y - Yhat
RR = np.power(RR, 2)
RR = RR.dot(RR)
RT = Y - Y.mean()
RT = np.power(RT, 2)
RT = RT.dot(RT)

R2 = 1 - RR/RT

print("R2 " + str(R2))

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()
