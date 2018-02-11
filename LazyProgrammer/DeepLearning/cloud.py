import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split

Nclass = 1000

X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

y = np.array([0] * Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = NeuralNetwork(hidden_layers=[2],
                      learning_rate = 10e-3,
                      epochs=1000,
                      activation='sigmoid')
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print "Traing score: ", train_score, " Test score: ", test_score
