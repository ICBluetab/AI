import numpy as np
import matplotlib.pyplot as plt

means = []

for i in range(1000):
    means.append(np.mean(np.random.uniform(-1, 1, 1000)))

plt.hist(means, bins=100)
plt.show()
