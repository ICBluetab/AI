import numpy as np
import matplotlib.pyplot as plt
import re

non_decimal = re.compile('[^\d]+')

X = []
Y = []

for line in open("moore.csv"):
    s = line.split('\t')

    x = int(non_decimal.sub('', s[2].split('[')[0]))
    y = int(non_decimal.sub('', s[1].split('[')[0]))

    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()

Y = np.log(Y)

plt.scatter(X, Y)
plt.show()

denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

Yhat = a * X + b

v1 = Y - Yhat
v2 = Y - Y.mean()

R2 = 1 - v1.dot(v1)/v2.dot(v2)

print("a " + str(a))
print("b " + str(b))
print("r - square " + str(R2))


plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# log(y) = log(a * x + b)
# y = exp(a * x ) * exp(b)
# 2 * y = 2 * exp(a * x) * exp(b)
# 2 * y = exp(ln(2)) * exp(a * x) * exp(b)
# 2 * Y = exp(a * x + ln(2)) * exp(b)

# exp(a * x2) * exp(b) = exp(a * x1 + ln(2)) * exp(b)
# a * x2 = a * x1 + ln(2)
# x2 = x1 + ln(2)/a

print("time to double: ", np.log(2)/a, " years")
