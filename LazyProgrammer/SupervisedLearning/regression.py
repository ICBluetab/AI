import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

N = 200

X = np.linspace(0, 10, N)
Y = np.sin(X)
X = X.reshape(N, 1)

plt.plot(X, Y)
plt.show()


Ntrain = 20
idx = np.random.choice(N, Ntrain)
Xtrain = X[idx]
Ytrain = Y[idx]

plt.scatter(Xtrain, Ytrain)
plt.plot(X, Y)
plt.show()


knn = KNeighborsRegressor(n_neighbors= 2)
knn.fit(Xtrain, Ytrain)
Yknn = knn.predict(X)

dt = DecisionTreeRegressor()
dt.fit(Xtrain, Ytrain)
Ydt = dt.predict(X)

plt.scatter(Xtrain, Ytrain)
plt.plot(X, Y)
plt.plot(X, Yknn, label='KNN')
plt.plot(X, Ydt, label='DT')
plt.legend()
plt.show()
