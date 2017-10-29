import numpy as np
import matplotlib.pyplot as plt

N=50
D=50

X = (np.random.random((N,D)) - 0.5) * 10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

print(true_w)

Y = X.dot(true_w) + np.random.randn(N) * 0.5

print(X)
print(Y)

cost = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 10.0
for t in xrange(5000):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate * (X.T.dot(delta) + l1 * np.sign(w))

    mse = delta.dot(delta)/np.sqrt(delta)
    cost.append(mse)

plt.plot(cost)
plt.show()

print("final w " + w)

plt.plot(true_w, label="true w")
plt.plo(w, label="w")
plt.lengend()
plt.show()
