import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

X =  np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

T = np.array([0, 1, 1, 0])


#plt.scatter(X[:,0], X[:,1], c=T)
#plt.show()

ones = np.array([[1]*N]).T
xy = np.array([X[:,0] * X[:,1]]).T
Xb = np.array(np.concatenate((ones, xy, X), axis = 1))

w = np.random.randn(D+2)

z = Xb.dot(w)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

Y = sigmoid(z)

#-(t.log(y) + (1-t)log(1-y))
def cross_entropy(T, Y):
    E = 0
    for i in xrange(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        elif T[i] == 0:
            E -= np.log(1 - Y[i])

    return E

learning_rate = 0.01
error = []
for i in xrange(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 100 == 0:
        print e

    w += learning_rate * (np.dot((T - Y).T, Xb) - 0.01*w)
    Y = sigmoid(Xb.dot(w))


plt.plot(error)
plt.title("Cross-entropy")

print "Final w: ", w
print "Final classification rate: ", (1 - np.abs(T - np.round(Y)).mean())
