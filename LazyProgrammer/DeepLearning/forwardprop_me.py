import numpy as np

N = 10
D = 5
Z = 3
K = 2

def target_matrix(T):
    TM = np.zeros((N, K))
    for i in xrange(N):
        TM[i, T[i]] = 1
    return TM

def argmax(T):
    That = np.zeros((N, K))
    m = np.argmax(T, axis=1)
    for i in xrange(N):
        That[i, m[i]] = 1
    return That

def accuracy(Y, T):
    return ((argmax(Y) == T).sum(axis=1) / K).sum() / float(N)

def tanh(T):
    eplus =  np.exp(T)
    eminus = np.exp(-T)
    return (eplus - eminus) / (eplus + eminus)

def softmax(T):
     a = np.exp(T)
     return a / np.sum(a, axis=1, keepdims=True)

X = np.random.random((N, D))
T = np.random.randint(0, high=K, size=N)
TM = target_matrix(T)

W = np.random.random((D, Z))
b = np.random.random(3)

V = np.random.random((Z, K))
c = np.random.random(K)

Y = softmax(tanh(X.dot(W) + b).dot(V) + c)

print("result ", Y)
print("accuracy ", accuracy(Y, TM))
