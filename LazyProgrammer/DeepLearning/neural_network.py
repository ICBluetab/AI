import numpy as np

class NeuralNetwork(object):
    def __init__(self, hidden_layers=[2], learning_rate = 10e-7,
                       epochs=10000, activation='relu'):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs

        if activation == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_d = self.sigmoid_d
        elif activation == 'tanh':
            self.activation = self.tanh
            self.activation_d = self.tanh_d
        else:
            self.activation = self.relu
            self.activation_d = self.relu_d

    def sigmoid(self, x, w):
        return 1 / (1 + np.exp(-x.dot(w)))

    def sigmoid_d(self, z):
        return (1 - z) * z

    def tanh(self, x, w):
        t = np.exp(2 * x.dot(w))
        return (t - 1)/ (t + 1)

    def tanh_d(self, z):
        return 1 - z * z

    def relu(self, x, w):
        t = x.dot(w)
        return t * (t > 0)

    def relu_d(self, z):
        return z > 0

    def forward(self, X, W):
        Z = [X]
        # forward
        for i in xrange(len(W) - 1):
            # activation
            Z.append(self.activation(Z[-1], W[i]))

        # softmax
        A = Z[-1].dot(W[-1])
        expA = np.exp(A)
        Y = expA / expA.sum(axis=1, keepdims=True)
        return Y, Z

    def one_hot_encoding(self, y, N, K):
        T = np.zeros((N, K))
        for i in xrange(N):
            T[i, y[i]] = 1
        return T

    def rescale(self, X):
        N, D = X.shape
        Xr = np.zeros((N, D))
        for d in xrange(D):
            Xr[:,d] = (X[:,d] - X[:,d].mean()) / X[:,d].std()

        return Xr

    def absorb_bias_term(slef, X):
        N = X.shape[0]
        ones = np.array([[1]*N]).T
        return np.concatenate((ones, X), axis=1)

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

    def softmax_d(self, T, Y, Z):
        dz = (T - Y)
        return Z[-1].T.dot(dz), dz

    def backprop_d(self, n, T, Y, Z, W, acc):
        dz = acc.dot(W[n+1].T) * self.activation_d(Z[n+1])
        return Z[n].T.dot(dz), dz

    def backprop(self, X, y, T, W):
        for epoch in xrange(self.epochs):
            Y_given_X, Z = self.forward(X, W)
            if epoch % 100 == 0:
                self.trace(y, T, Y_given_X)

            D, acc = self.softmax_d(T, Y_given_X, Z)
            W[-1] += self.learning_rate * D
            for i in reversed(xrange(len(W) - 1)):
                D, acc =  self.backprop_d(i, T, Y_given_X, Z, W, acc)
                W[i] += self.learning_rate * D

    def fit(self, X, y):
        # D: number of features
        N, D = X.shape
        # K number of outputs. From 0 to K -1
        K = np.max(y) + 1

        Xr = self.rescale(X)
        Xb = self.absorb_bias_term(Xr)

        self.W = []
        node_distribution = [D + 1] + self.hidden_layers + [K]
        for i in xrange(len(node_distribution) - 1):
            self.W.append(np.random.randn(node_distribution[i],
                                          node_distribution[i + 1]))

        T = self.one_hot_encoding(y, N, K)
        self.backprop(Xb, y, T, self.W)


    def predict(self, X):
        Xr = self.rescale(X)
        Xb = self.absorb_bias_term(Xr)
        P, _ = self.forward(Xb, self.W)
        return P

    def score(self, X, y):
        P = self.predict(X)
        return self.classification_rate(y, P)
