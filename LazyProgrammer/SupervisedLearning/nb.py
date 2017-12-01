import numpy as np
from util import get_data
from datetime import datetime
from scipy.stats import multivariate_normal as mvn



class NBModel(object):
    def __init__(self, classes):
        self.classes = classes

    def fit(self, X, Y, smoothing=10e-3):
        self.dict_of_gaussians = {}
        self.priors = {}
        for c in self.classes:
            Xc = X[Y == c]
            mu = Xc.mean(axis=0)
            var = Xc.var(axis=0) + smoothing
            self.dict_of_gaussians[c] = {'mu': mu, 'var': var}
            self.priors[c] = float(len(Xc)) / float(len(X))

    def predict(self, X):
        N, D = X.shape
        K = len(self.dict_of_gaussians)
        P = np.zeros((N,K))
        for c in self.classes:
            g = self.dict_of_gaussians[c]
            mu, var = g['mu'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mu, cov=var) + np.log(self.priors[c])

        return np.argmax(P, axis = 1)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

def main():
    X, Y = get_data(10000)
    Ntrain = len(X)/2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest   = X[Ntrain:], Y[Ntrain:]
    t0 = datetime.now()
    model = NBModel(set(Y))
    model.fit(Xtrain, Ytrain)
    print "Training time: ", (datetime.now() - t0)

    t0 = datetime.now()
    print "Train accuracy: ", model.score(Xtrain, Ytrain), " Train size ", len(Ytrain)
    print "Prediction time: ", (datetime.now() - t0)

    t0 = datetime.now()
    print "Test accuracy: ", model.score(Xtest, Ytest), " Test size ", len(Ytest)
    print "Prediction time: ", (datetime.now() - t0)

if __name__ == '__main__':
    main()
