import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split

X = np.zeros((1000, 2))
X[:250]     = np.random.random((250, 2)) / 2 + 0.5
X[250:500]  = np.random.random((250, 2)) / 2
X[500:750] = np.random.random((250, 2)) / 2 + np.array([[0, 0.5]])
X[750:1000] = np.random.random((250, 2)) / 2 + np.array([[0.5, 0]])
y = np.array([0] * 500 + [1] * 500)


plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = NeuralNetwork(hidden_layers=[20, 20, 20, 20],
                      learning_rate = 0.00000005,
                      epochs=10000, activation='relu')
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print "Traing score: ", train_score, " Test score: ", test_score
