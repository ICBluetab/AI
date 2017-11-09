import numpy as np
import pandas as pd

def get_data():
    df = pd.read_csv("ecommerce_data.csv")
    data = df.as_matrix()

    X = data[:, :-1]
    Y = data[:, -1]

    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()

    N, D = X.shape
    X2 = np.zeros((N, D + 3))
    X2[:, :(D -1)] = X[:, :(D -1)]

    #for n in xrange(N):
    #    t = int(X[n, D-1])
    #    X2[n, t + D - 1] = 1

    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    X2[:, -4:] = Z

    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2



get_binary_data()
