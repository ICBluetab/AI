import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def getBinaryData():
    X = []
    Y = []

    first = True
    for line in open("../../LargeFiles/fer2013.csv"):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_cost(T, Y):
    return - (T * np.log(Y) + (1- T) * np.log(1 - Y)).sum()

def error_rate(T, P):
    return np.mean(T != P)


class LogisticModel(object):

    def __init__(self):
        pass

    def fit(self, X, Y, learning_rate=10.e-7, reg=0, epochs=120000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        self.W = np.random.randn(D) / np.sqrt(D)
        self.b = 0

        costs = []
        best_validation_error = 1
        for i in xrange(epochs):
            pY = self.forward(X)

            self.W -= learning_rate * (X.T.dot(pY - Y) + reg*self.W)
            self.b -= learning_rate * ((pY - Y).sum() + reg*self.b)

            if i % 20 == 0:
                pYvalid = self.forward(Xvalid)
                c = sigmoid_cost(Yvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.round(pYvalid))
                print "i: ", i, " cost ", c, " error: ", e
                if e < best_validation_error:
                    best_validation_error = e

        print "best_validation_error: ", best_validation_error

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        return sigmoid(X.dot(self.W) + self.b)

    def predict(self, X):
        P = self.forward(X)
        return np.round(P)

    def score(self, X, Y):
        P = self.predict(X)
        return 1 - error_rate(Y, P)

def main():
    X, Y = getBinaryData()

    X0 = X[Y==0, : ]
    X1 = X[Y==1, :]
    X1 = np.repeat(X1, 9, axis= 0)
    X = np.vstack([X0, X1])
    Y = np.array([0]*len(X0) + [1]*len(X1))

    model = LogisticModel()
    model.fit(X, Y, show_fig=True)
    model.score(X, Y)

if __name__ == '__main__':
    main()
