import numpy as np
import matplotlib.pyplot as plt

i = np.random.randn(5000, 2)
f = np.linalg.norm(i, axis=1)
i = i[(f > 2) & (f < 2.3)]

plt.scatter(i[:, 0], i[:, 1], c='blue')

o = 2 * np.random.randn(10000, 2)
f = np.linalg.norm(o, axis=1)
o = o[(f > 4) & (f < 4.5)]

plt.scatter(o[:, 0], o[:, 1], c='red')

plt.axis("equal")
plt.show()
