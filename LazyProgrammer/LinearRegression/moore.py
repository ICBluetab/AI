import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

p = re.compile('(\d\d\d\d)')

def year_to_numeric(year):
    return np.float(year)

def value_to_numeric(value):
    return np.float(value.replace(',', ''))

df = pd.read_csv("moore.csv", header=None, sep='\t')
df[1] = df[1].apply(value_to_numeric)
df[2] = df[2].apply(year_to_numeric)

X = df[2].values
Y = df[1].values

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator


Yhat = a * X + b

v1 = Y - Yhat
v2 = Y - Y.mean()

R2 = 1 - v1.dot(v1)/v2.dot(v2)

print("R2 " + str(R2))


plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()
