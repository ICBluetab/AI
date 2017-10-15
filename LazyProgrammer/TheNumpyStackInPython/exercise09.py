import numpy as np
import pandas as pd

i = np.random.randn(5000, 2)
f = np.linalg.norm(i, axis=1)
i = i[(f > 2) & (f < 2.3)]

df_i = pd.DataFrame(i, columns = ['x1', 'x2'])
df_i['y'] = 0

o = 2 * np.random.randn(10000, 2)
f = np.linalg.norm(o, axis=1)
o = o[(f > 4) & (f < 4.5)]

df_o = pd.DataFrame(o, columns = ['x1', 'x2'])
df_o['y'] = 1

pd.concat([df_i, df_o]).to_csv("exercise09.csv", index=False)
