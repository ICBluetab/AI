import numpy as np

class NeuralNetwork(object):
    def __init__(self, hidden_layers=1, hidden_units=2, learning_rate = 10e-7,
                    ephocs=10000000):
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.ephocs = ephocs

    def forward(self, X):
        Z0 = 1 / (1 + np.exp(-(X.dot(self.W0) + self.b0))) # sigmoid
        A = Z0.dot(self.W1) + self.b1
        expA = np.exp(A)
        y = expA / expA.sum(axis=1, keepdims=True) # softmax
        return y, Z0

    def one_hot_encoding(self, y, N, K):
        T = np.zeros((N, K))
        for i in xrange(N):
            T[i, y[i]] = 1
        return T

    def derivative_w1(self, Z0, T, O):
        N, K = O.shape
        M = Z0.shape[1]
        ret = Z0.T.dot(T - O)
        return ret

    def derivative_b1(self, T, O):
        return (T - O).sum(axis=0)

    def derivative_w0(self, X, Z0, T, O):
        N, D = X.shape
        M, K = self.W1.shape
        dZ = (T - O).dot(self.W0.T) * Z0 * (1 - Z0) # sigmodi
        return X.T.dot(dZ)

    def derivative_b0(self, T, O, Z0):
        return ((T - O).dot(self.W1.T) * Z0 * (1 - Z0)).sum(axis=0) # sigmoid

    def backprop(self, X, y):
        N, D = X.shape # D number of features
        K = np.max(y) + 1 # K number of outputs. From 0 to K -1

        T = self.one_hot_encoding(y, N, K)

        self.W0 = np.random.randn(D, self.hidden_units)
        self.b0 = np.random.randn(self.hidden_units)
        self.W1 = np.random.randn(self.hidden_units, K)
        self.b1 = np.random.randn(K)

        for epoch in xrange(self.ephocs):
            o, Z0 = self.forward(X)
            self.W1 += self.learning_rate * self.derivative_w1(Z0, T, o)
            self.b1 += self.learning_rate * self.derivative_b1(T, o)
            self.W0 += self.learning_rate * self.derivative_w0(X, Z0, T, o)
            self.b0 += self.learning_rate * self.derivative_b0(T, o, Z0)

    def fit(self, X, y):
        self.backprop(X, y)

    def predict(self, X):
        P, Z0 = self.forward(X)
        return P

    def score(self, X, y):
        p = np.argmax(self.predict(X), axis=1)
        N = len(y)
        n_correct = 0
        for i in xrange(N):
            if y[i] == p[i]:
                n_correct += 1

        return float(n_correct) / N
