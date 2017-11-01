import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


F = [['X2', 'Doctor availability per 100,000 residents'],
     ['X3', 'Hospital availability per 100,000 residents'],
     ['X4', 'Annual per capita income in thousands of dollars'],
     ['X5', 'Population density people per square mile']]

L = 'X1'

df = pd.read_excel("health.xls")

# Coeficione of determination
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

# X1 = death rate per 1000 residents
# X2 = doctor availability per 100,000 residents
# X3 = hospital availability per 100,000 residents
# X4 = annual per capita income in thousands of dollars
# X5 = population density people per square mile

def scatter_data(c1, c2):
    M = df[[c1, c2]].as_matrix()

    X = np.array(M[:,0]) # this is a numpy array
    Y = np.array(M[:,1]) # this is a numpy array

    plt.scatter(X, Y)
    plt.show()


#scatter_data('X1','X2')
#catter_data('X1','X3')
#scatter_data('X1','X4')
#scatter_data('X1','X5')

#scatter_data('X2','X1')
#scatter_data('X2','X3')
#scatter_data('X2','X4')
#scatter_data('X2','X5')

#scatter_data('X3','X1')
#scatter_data('X3','X2')
#scatter_data('X3','X4')
#scatter_data('X3','X5')

#scatter_data('X4','X1')
#scatter_data('X4','X2')
#scatter_data('X4','X3')
#scatter_data('X4','X5')

#scatter_data('X5','X1')
#scatter_data('X5','X2')
#scatter_data('X5','X3')
#scatter_data('X5','X4')

def univariate_lr(f):
    M = df[[f[0],L]].as_matrix()

    X = np.array(M[:,0]) # this is a numpy array
    Y = np.array(M[:,1]) # this is a numpy array

    X = np.vstack([np.ones(len(X)), X]).T

    w = solve(X, Y)
    Y_hat = X.dot(w)

    plt.scatter(X[:,1], Y)
    plt.plot(X, Y_hat)
    plt.xlabel(f[1])
    plt.ylabel("Death rate per 1000 residents")
    plt.legend()
    plt.show()

    print("r - square " + f[0] + " " + str(rsqrt(Y, Y_hat)))

#for f in F:
#    univariate_lr(f)

def multivariate_lr():
    M = df[['X2', 'X3', 'X4', 'X5', L]].as_matrix()

    X = np.array(M[:,:4]) # this is a numpy array
    Y = np.array(M[:,4]) # this is a numpy array

    X = np.insert(X, 0, 1, axis=1)

    w = solve(X, Y)
    Y_hat = X.dot(w)

    print("--- Lineal Regression ---")
    print("w " + str(w))
    print("r - square " + str(rsqrt(Y, Y_hat)))
    print("mse " + str(mse(Y, Y_hat)))

multivariate_lr()
