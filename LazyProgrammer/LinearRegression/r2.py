import numpy as np

X = []
Y = []
for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append([1, float(x)])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

def get_r2(A, B):
    w = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, B))
    Bhat = np.dot(A, w)

    v1 = B - Bhat
    v2 = B - B.mean()
    return 1 - v1.dot(v1)/v2.dot(v2)


print "r - square: ", get_r2(X, Y)

n = len(X)
r = np.random.rand(n).reshape(n, 1)
X = np.append(X, r, axis = 1)

print "r - square: ", get_r2(X, Y)
