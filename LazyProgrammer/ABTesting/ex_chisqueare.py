import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2


def get_p_value(T):
    det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
    c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:, 0].sum() / T[:, 1].sum()
    p = 1 - chi2.cdf(x=c2, df=1)
    return p

df = pd.read_csv('advertisement_clicks.csv')
a = df[df['advertisement_id'] == 'A']
b = df[df['advertisement_id'] == 'B']
a = a['action'].as_matrix()
b = b['action'].as_matrix()

T = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])

p = get_p_value(T)

print(p)
