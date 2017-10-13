import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../LargeFiles/train.csv")
df = df.loc[df['label'] == 7]
df.drop('label', axis=1, inplace=True)
img = df.mean().as_matrix().reshape(28,28)
plt.imshow(255 - img, cmap='gray')
plt.show()
