import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel("mlr02.xls")
X = df.as_matrix()

plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]
X2Only = df[['X2', 'ones']]
X3Only = df[['X3', 'ones']]

def get_r2(X, Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Yhat = np.dot(X, w)

    v1 = Y - Yhat
    v2 = Y - Y.mean()
    return 1 - v1.dot(v1)/v2.dot(v2)


print "r2 x2 only: ", get_r2(X2Only, Y)
print "r2 x3 only: ", get_r2(X3Only, Y)
print "r2 for both: ", get_r2(X, Y)
