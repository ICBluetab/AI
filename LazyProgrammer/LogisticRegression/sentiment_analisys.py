import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('sentiment_words.csv.out')
M = df.as_matrix()

M = np.round(M)

X = M[:,:-1]
Y = M[:,-1]

X, Y = shuffle(X, Y)

Xtrain = X[:-100]
Ytrain = Y[:-100]

Xtest = X[-100:]
Ytest = Y[-100:]

N = Xtrain.shape[0]
D = Xtrain.shape[1]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#-(t.log(y) + (1-t)log(1-y))
def classification_rate(T, Y):
    return np.mean(T == Y)

def cross_entropy(T, Y):
    return - np.mean(T*np.log(Y) + (1 -T)*np.log(1- Y))

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, Xtrain), axis=1)

w = np.random.randn(D+1)/np.sqrt(D+1)

learning_rate = 0.00000001
l1 = 0.01
for i in xrange(120000):
    Yhat = sigmoid(Xb.dot(w))
    if i % 100 == 0:
        print "cross_entropy ", cross_entropy(Ytrain, Yhat)
    w = w - learning_rate * (Xb.T.dot(Yhat - Ytrain) + l1 * np.sign(w))

print "classification_rate ", classification_rate(Ytrain, np.round(Yhat))
print "W : " , w

print "Prediction"

N = Xtest.shape[0]
D = Xtest.shape[1]

ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, Xtest), axis=1)

P = sigmoid(Xb.dot(w))
print "cross_entropy ", cross_entropy(Ytest, P)
print "clasisification_rate ", classification_rate(Ytest, np.round(P))
