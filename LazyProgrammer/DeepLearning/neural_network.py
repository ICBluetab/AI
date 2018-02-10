import numpy as np

class NeuralNetwork(object):
    def __init__(self, hidden_layers=1, hidden_units=[2], learning_rate = 10e-7,
                    ephocs=1000):
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.ephocs = ephocs

    def forward(self, X):
        Z = []
        X_hat = X
        for i in range(self.hidden_layers):
            X_hat = 1 / (1 + np.exp(-(X_hat.dot(self.W[i]) + self.b[i]))) # sigmoid
            Z.append(X_hat)

        A = X_hat.dot(self.W[-1]) + self.b[-1]
        expA = np.exp(A)
        y = expA / expA.sum(axis=1, keepdims=True) # softmax
        return y, Z

    def one_hot_encoding(self, y, N, K):
        T = np.zeros((N, K))
        for i in xrange(N):
            T[i, y[i]] = 1
        return T

    def softmax_derivative_w(self, Z, T, O):
        N, K = O.shape
        M = Z[-1].shape[1]
        ret = Z[0].T.dot(T - O)
        return ret

    def softmax_derivative_b(self, T, O):
        return (T - O).sum(axis=0)

    def sigmoid_derivative_w(self, X, Z, T, O):
        N, D = X.shape
        M, K = self.W[1].shape
        dZ = (T - O).dot(self.W[0].T) * Z[0] * (1 - Z[0])
        return X.T.dot(dZ)

    def sigmoid_derivative_b(self, Z, T, O):
        return ((T - O).dot(self.W[1].T) * Z[0] * (1 - Z[0])).sum(axis=0)

    def derivative_w1(self, Z, T, O):
        N, K = O.shape
        M = Z[0].shape[1]
        ret = Z[0].T.dot(T - O)
        return ret

    def derivative_b1(self, T, O):
        return (T - O).sum(axis=0)

    def derivative_w0(self, X, Z, T, O):
        N, D = X.shape
        M, K = self.W[1].shape
        dZ = (T - O).dot(self.W[0].T) * Z[0] * (1 - Z[0]
        ) # sigmoid
        return X.T.dot(dZ)

    def derivative_b0(self, T, O, Z):
        return ((T - O).dot(self.W[1].T) * Z[0] * (1 - Z[0])).sum(axis=0) # sigmoid

    def derivative_w(self, i,  T, O):
        dZ = (T - O).dot(self.W[i].T) * Z[i] * (1 - Z[i]) # sigmoid
        return Z[i - 1].T.dot(dZ)

    def derivative_b(self, i, T, O):
        return ((T - O).dot(self.W[i + 1].T) * Z[i] * (1 - Z[i])).sum(axis=0) # sigmoid

    def backprop(self, X, y):
        N, D = X.shape # D number of features
        K = np.max(y) + 1 # K number of outputs. From 0 to K -1

        self.W = []
        self.b = []

        self.W.append(np.random.randn(D, self.hidden_units[0]))
        self.b.append(np.random.randn(self.hidden_units[0]))

        for w in xrange(self.hidden_layers - 1):
            self.W.append(np.random.randn(self.hidden_units[w], self.hidden_units[w + 1]))
            self.b.append(np.random.randn(self.hidden_units[w + 1]))

        self.W.append(np.random.randn(self.hidden_units[-1], K))
        self.b.append(np.random.randn(K))

        T = self.one_hot_encoding(y, N, K)


        for epoch in xrange(self.ephocs):
            O, Z = self.forward(X)
            self.W[-1] += self.learning_rate * self.softmax_derivative_w(Z[-1], T, O)
            self.b[-1] += self.learning_rate * self.softmax_derivative_b(T, O)

            for i in reversed(xrange(self.hidden_layers -  1))
                self.W[i] += self.learning_rate * self.derivative_w0(X, Z, T, O)
                self.b[i] += self.learning_rate * self.derivative_b0(Z, T, O)

            self.W[0] += self.learning_rate * self.derivative_w0(X, Z, T, O)
            self.b[0] += self.learning_rate * self.derivative_b0(Z, T, O)


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
