import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_data():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300, 2))*2 - 1
    Y = np.sign(X.dot(w) + b)
    return X, Y


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
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()

    Ntrain = len(Y)/ 2

    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = PercentronModel()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print "Training time: ", (datetime.now() - t0)

    t0 = datetime.now()
    print "Train score: ", model.score(Xtrain, Ytrain)
    print "Prediction time: ", (datetime.now() - t0), "Train size: ",  len(Ytrain)

    t0 = datetime.now()
    print "Test score: ", model.score(Xtest, Ytest)
    print "Prediction time: ", (datetime.now() - t0) , "Test size: ",  len(Ytest)
