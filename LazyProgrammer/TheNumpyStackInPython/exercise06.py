import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = np.random.random((500, 2))

print(np.logical_xor(d , d))

plt.scatter(d[:,0], d[:,1], c='blue')
plt.scatter(d[:,0] - 1, d[:,1] - 1, c='blue')
plt.scatter(d[:,0] -1 , d[:,1], c='red')
plt.scatter(d[:,0], d[:,1] - 1, c='red')
plt.show()
