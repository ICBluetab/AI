import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../LargeFiles/train.csv")
df = df.loc[df['label'] == 7]
df.drop('label', axis=1, inplace=True)
img = df.iloc[0].as_matrix().reshape(28,28)
img = np.rot90(img, k=1, axes=(1, 0))
plt.imshow(255 - img, cmap='gray')
plt.show()
