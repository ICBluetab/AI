import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split

N = 1000
D = 2

R_inner = 5
R_outer = 10

R1 = np.random.randn(N/2) + R_inner
theta = 2*np.pi*np.random.random(N/2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N/2) + R_outer
theta = 2*np.pi*np.random.random(N/2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([X_inner, X_outer])
y = np.array([0] * (N/2) + [1] * (N/2))

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = NeuralNetwork()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print "Traing score: ", train_score, " Test score: ", test_score
