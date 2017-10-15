import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0.3, 0.6, 0.1], [0.5, 0.2, 0.3], [0.4, 0.1, 0.5]])

v = np.array([1/3., 1/3., 1/3.])

d = []

for i in range(25):
    vi = v.dot(A)
    d.append(np.linalg.norm(vi - v))
    v = vi

plt.plot(d)

plt.show()
