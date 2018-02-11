import numpy as np

class NeuralNetwork(object):
    def __init__(self, hidden_layers=[2], learning_rate = 10e-7,
                    epochs=10000):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

    def forward(self, X, W, B):
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

    def one_hot_encoding(self, y, N, K):
        T = np.zeros((N, K))
        for i in xrange(N):
            T[i, y[i]] = 1
        return T

    def derivative(self, n, T, Y, Z, W, B):
        D = (T - Y)
        l = len(W) -1
        while l > n:
            D = D.dot(W[l].T) * Z[l] * (1 - Z[l])
            l -= 1

        return Z[n].T.dot(D), D.sum(axis=0)

    def cost(self, T, y):
        tot = T * np.log(y)
        return tot.sum()

    def classification_rate(self, y, P):
        p = np.argmax(P, axis=1)
        n_correct = 0
        N = len(y)
        n_correct = 0
        for i in xrange(N):
            if y[i] == p[i]:
                n_correct += 1
        return float(n_correct) / N

    def trace(self, y, T, Y_given_X):
        c = self.cost(T, Y_given_X)
        r = self.classification_rate(y, Y_given_X)
        print "costs: ", c, "classification_rate: ", r

    def backprop(self, X, y, T, W, B):
        for epoch in xrange(self.epochs):
            Y_given_X, Z = self.forward(X, W, B)
            if epoch % 100 == 0:
                self.trace(y, T, Y_given_X)

            for i in reversed(xrange(len(W))):
                dw, db = self.derivative(i, T, Y_given_X, Z, W, B)
                W[i] += self.learning_rate * dw
                B[i] += self.learning_rate * db


    def fit(self, X, y):
        N, D = X.shape # D number of features
        K = np.max(y) + 1 # K number of outputs. From 0 to K -1

        W = []
        B = []

        node_distribution = [D] + self.hidden_layers + [K]
        for i in xrange(len(node_distribution) - 1):
            W.append(np.random.randn(node_distribution[i],
                                          node_distribution[i + 1]))
            B.append(np.random.randn(node_distribution[i + 1]))

        self.W = W
        self.B = B

        T = self.one_hot_encoding(y, N, K)
        self.backprop(X, y, T, W, B)


    def predict(self, X):
        P, _ = self.forward(X, self.W, self.B)
        return P

    def score(self, X, y):
        P = self.predict(X)
        return self.classification_rate(y, P)
