import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

X[:50, :] = X[:50, :] - 2*np.ones((50, D))
X[50:, :] = X[50:, :] + 2*np.ones((50, D))

T = np.array([0]*50 + [1]*50)

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

w = np.random.randn(D + 1)

z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

#-(t.log(y) + (1-t)log(1-y))
def cross_entropy(T, Y):
    E = 0
    for i in xrange(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        elif T[i] == 0:
            E -= np.log(1 - Y[i])

    return E

learning_rate = 0.1
for i in xrange(100):
    if i % 10 == 0:
        print cross_entropy(T, Y)

    w += learning_rate * (T - Y).dot(Xb)
    Y = sigmoid(Xb.dot(w))

print "Final w: ", w
