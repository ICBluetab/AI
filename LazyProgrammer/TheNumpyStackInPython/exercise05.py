import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def isSymetric(m):
    return (m.T == m).all()

S = np.array([[1, 2], [2, 1]])

print(isSymetric(S))

X = np.array([[1, 3], [2, 1]])

print(isSymetric(X))
