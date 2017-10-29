import numpy as np
import matplotlib.pyplot as plt

N = 50
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)

Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

X =  np.vstack([np.ones(N), X]).T

w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_ml = X.dot(w_ml)

#plt.scatter(X[:,1], Y)
#plt.plot(X[:,1], Y_ml)
#plt.show()

l2 = 1000.0
w_map = np.linalg.solve(l2 * np.eye(2) + X.T.dot(X), X.T.dot(Y) )
Y_map = X.dot(w_map)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_ml, 'C1')
plt.plot(X[:,1], Y_map, 'C2')
plt.show()
