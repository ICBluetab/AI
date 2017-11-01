import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Coeficient of determination
def rsqrt(Y, Y_hat):
    r1 = Y - Y_hat
    r2 = Y - Y.mean()
    return 1 - r1.dot(r1)/r2.dot(r2)

# Mean squared error
def mse(Y, Y_hat):
    r1 = Y - Y_hat
    return r1.dot(r1) / len(r1)

def solve(X, Y) :
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def solve_l2(X, Y, l2) :
    return np.linalg.solve(l2 * np.eye(2) +  X.T.dot(X), X.T.dot(Y))

def solve_gd(X, Y) :
    w = np.random.randn(X.shape[1])/np.sqrt(X.shape[1])
    for i in xrange(5000):
        Y_hat = X.dot(w)
        w = w - 0.001 * X.T.dot(Y_hat - Y)
    return w

def solve_l2_gd(X, Y, l2) :
    w = np.random.randn(X.shape[1])/np.sqrt(X.shape[1])
    for i in xrange(5000):
        Y_hat = X.dot(w)
        w = w - 0.001 * (X.T.dot(Y_hat - Y) + l2 * w)

    return w

df = pd.read_excel("unemployment.xls")

# X = national unemployment rate for adult males
# Y = national unemployment rate for adult females
M = df[['X', 'Y']].as_matrix()

X = np.array(M[:,0])
X = np.vstack([np.ones(len(X)), X]).T
Y = np.array(M[:,1])

w = solve(X, Y)
Y_hat = X.dot(w)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_hat, 'b')

print("--- Lineal Regression ---")
print("r - square " + str(rsqrt(Y, Y_hat)))
print("mse " + str(mse(Y, Y_hat)))

w = solve_l2(X, Y, 1) # No merora absolutamente nada
Y_hat_l2 = X.dot(w)

plt.plot(X[:,1], Y_hat_l2, 'r')
plt.xlabel("National unemployment rate for adult males")
plt.ylabel("National unemployment rate for adult females")
plt.show()


print("--- L2 regularization ---")
print("r - square " + str(rsqrt(Y, Y_hat_l2)))
print("mse " + str(mse(Y, Y_hat_l2)))

w = solve_gd(X, Y)
Y_hat = X.dot(w)

plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Y_hat, 'b')

print("--- Lineal Regression Gradiend Descent ---")
print("r - square " + str(rsqrt(Y, Y_hat)))
print("mse " + str(mse(Y, Y_hat)))

w = solve_l2_gd(X, Y, 1) # No merora absolutamente nada
Y_hat_l2 = X.dot(w)

plt.plot(X[:,1], Y_hat_l2, 'r')
plt.xlabel("National unemployment rate for adult males")
plt.ylabel("National unemployment rate for adult females")
plt.show()


print("--- L2 regularization Gradient Descent ---")
print("r - square " + str(rsqrt(Y, Y_hat_l2)))
print("mse " + str(mse(Y, Y_hat_l2)))
