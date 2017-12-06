import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from util import get_data


class PercentronModel(object):
    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=1.0, epochs=1000):
        N, D = X.shape
        self.W = np.random.randn(D)
        self.b = 0

        costs = []
        for i in xrange(epochs):
            P = self.predict(X)
            incorrect = np.nonzero(Y != P)[0]
            if len(incorrect) == 0:
                break
            idx = np.random.choice(incorrect)
            self.W = self.W + learning_rate * Y[idx] * X[idx]
            self.b = self.b + learning_rate * Y[idx]

            c = len(incorrect) / float(N)
            costs.append(c)

        print "finsal w: ", self.W, "final b: ", self.b, "epochs: ", (i + 1), "/", epochs

        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.sign(X.dot(self.W) + self.b)


    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == '__main__':
    X, Y = get_data()
    idx = np.logical_or( Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    Y[Y == 0] = -1

    model = PercentronModel()
    t0 = datetime.now()
    model.fit(X, Y, learning_rate=10e-3)
    print "Training time: ", (datetime.now() - t0)
    print "Train score: ", model.score(X, Y)
