import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel("health.xls")

def solve(X, Y) :
    pass



# X1 = death rate per 1000 residents
# X2 = doctor availability per 100,000 residents
# X3 = hospital availability per 100,000 residents
# X4 = annual per capita income in thousands of dollars
# X5 = population density people per square mile

def scatter_data(column):
    M = df[[column,'X1']].as_matrix()

    X = np.array(M[:,0]) # this is a numpy array
    Y = np.array(M[:,1]) # this is a numpy array

    plt.scatter(X, Y)
    plt.show()


#scatter_data('X2')
#scatter_data('X3')
#scatter_data('X4')
#scatter_data('X5')
#scatter_data('X6')

def linear_regression(column):
    M = df[[column,'X1']].as_matrix()

    X = np.array(M[:,0]) # this is a numpy array
    Y = np.array(M[:,1]) # this is a numpy array

    X = np.vstack([np.ones(len(X)), X]).T
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    print(w)

linear_regression('X2')
