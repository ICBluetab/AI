import numpy as np
import matplotlib.pyplot as plt

def forward(X, w0, b0, w1, b1, w2, b2):
    Z0 = 1 / (1 + np.exp(-X.dot(w0) - b0))
    Z1 = 1 / (1 + np.exp(-Z0.dot(w1) - b1))
    A = Z1.dot(w2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z0, Z1

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1

    return float(n_correct) / n_total


def derivative_w2(Z1, T, Y):
    N, K = T.shape
    M = Z1.shape[1]
    return Z1.T.dot(T - Y)

def derivative_b2(T, Y):
    return (T - Y).sum(axis=0)

def derivative_w1(Z0, Z1, T, Y, w2):
    dZ1 = (T - Y).dot(w2.T) * Z1 * (1 - Z1)
    return Z0.T.dot(dZ1)

def derivative_b1(Z1, T, Y, w2):
    dZ1 = (T - Y).dot(w2.T) * Z1 * (1 - Z1)
    return dZ1.sum(axis=0)

def derivative_w0(X, Z0, Z1, T, Y, w1, w2):
    dZ1 = (T - Y).dot(w2.T) * Z1 * (1 - Z1)
    dZ0 = dZ1.dot(w1.T) * Z0 * (1 - Z0)
    return X.T.dot(dZ0)

def derivative_b0(Z0, Z1, T, Y, w1, w2):
    dZ1 = (T - Y).dot(w2.T) * Z1 * (1 - Z1)
    dZ0 = dZ1.dot(w1.T) * Z0 * (1 - Z0)
    return dZ0.sum(axis=0)

def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()

def main():
    Nclass = 500
    D = 2
    S = 3
    M = 3
    K = 3

    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0] * Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)

    T = np.zeros((N, K)) # one hot encoding
    for i in xrange(N):
        T[i, Y[i]] = 1

    plt.scatter(X[:,0], X[:,1], c=Y,s=100, alpha=0.5)
    plt.show()

    w0 = np.random.randn(D, S)
    b0 = np.random.randn(S)
    w1 = np.random.randn(S, M)
    b1 = np.random.randn(M)
    w2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 10e-7
    costs = []

    for epoch in xrange(1000):
        output, Z0, Z1 = forward(X, w0, b0, w1, b1, w2, b2)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print "costs: ", c, "classification_rate: ", r
            costs.append(c)

        w2 += learning_rate * derivative_w2(Z1, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        w1 += learning_rate * derivative_w1(Z0, Z1, T, output, w2)
        b1 += learning_rate * derivative_b1(Z1, T, output, w2)
        w0 += learning_rate * derivative_w0(X, Z0, Z1, T, output, w1, w2)
        b0 += learning_rate * derivative_b0(Z0, Z1, T, output, w1, w2)

    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
