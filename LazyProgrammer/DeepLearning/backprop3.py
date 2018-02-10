import numpy as np
import matplotlib.pyplot as plt

def forward(X, W, b0, w1, b1, w2, b2):
    Z0 = 1 / (1 + np.exp(-X.dot(w0) - b0))
    Z1 = 1 / (1 + np.exp(-Z0.dot(w1) - b1))
    A = Z1.dot(w2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z0, Z1

def forward(X, W, B):
    Z = [X]
    # forward
    for i in xrange(len(W) - 1):
        # sigmoid
        Z.append(1 / (1 + np.exp(-Z[-1].dot(W[i]) - B[i])))

    # softmax
    A = Z[-1].dot(W[-1]) + B[-1]
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1

    return float(n_correct) / n_total

def derivative_W(n, T, Y, Z, W):
    D = (T - Y)
    l = len(W) -1
    while l > n:
        D = D.dot(W[l].T) * Z[l] * (1 - Z[l])
        l -= 1
    return Z[n].T.dot(D)

def derivative_B(n, T, Y, Z, W):
    D = (T - Y)
    l = len(W) -1
    while l > n:
        D = D.dot(W[l].T) * Z[l] * (1 - Z[l])
        l -= 1
    return D.sum(axis=0)


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

    W = [w0, w1, w2]
    B = [b0, b1, b2]

    learning_rate = 10e-7
    costs = []

    for epoch in xrange(100000):
        Y_given_X, Z = forward(X, W, B)
        if epoch % 100 == 0:
            c = cost(T, Y_given_X)
            P = np.argmax(Y_given_X, axis=1)
            r = classification_rate(Y, P)
            print "costs: ", c, "classification_rate: ", r
            costs.append(c)

        for i in reversed(xrange(len(W))):
            W[i] += learning_rate * derivative_W(i, T, Y_given_X, Z, W)
            B[i] += learning_rate * derivative_B(i, T, Y_given_X, Z, W)

    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
