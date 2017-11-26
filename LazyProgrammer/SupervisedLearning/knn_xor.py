import numpy as np
import matplotlib.pyplot as plt
from knn import KNN
from sklearn.utils import shuffle
from util import get_xor

if __name__ == '__main__':
    X, Y = get_xor()

    plt.scatter(X[:,0], X[:,1], c=Y)
    plt.show()

    model = KNN(3)
    model.fit(X, Y)
    print "Accuracy; ", model.score(X, Y)
