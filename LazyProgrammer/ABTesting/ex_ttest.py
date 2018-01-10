import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


df = pd.read_csv('advertisement_clicks.csv')
a = df[df['advertisement_id'] == 'A']
b = df[df['advertisement_id'] == 'B']
a = a['action'].as_matrix()
b = b['action'].as_matrix()

print("len(a)", len(a))
print("len(b)", len(b))

print("a.mean:", a.mean())
print("b.mean:", b.mean())

t2, p2 = stats.ttest_ind(a, b)
print "t:\t", t2, "p:\t", p2
