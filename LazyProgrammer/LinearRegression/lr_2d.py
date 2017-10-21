import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:,1], X[:,2], Y)
plt.show()

W = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat = X.dot(W)

v1 = Y - Yhat
v2 = Y - Y.mean()

R2 = 1 - v1.dot(v1)/v2.dot(v2)
print("r - square: "  + str(R2))


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:,1], X[:,2], Y, c='r')
ax.plot_trisurf(X[:,1], X[:,2], Yhat)
plt.show()
