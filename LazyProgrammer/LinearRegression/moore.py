import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

p = re.compile('(\d\d\d\d)')

def year_to_numeric(year):
    return np.float(year)

def value_to_numeric(value):
    return np.float(value.replace(',', ''))

df = pd.read_csv("moore.csv", header=None, sep='\t')
df[1] = df[1].apply(value_to_numeric)
df[2] = df[2].apply(year_to_numeric)

x = df[2].values
y = df[1].values

xy_ = np.mean(x * y)
x_y_ = np.mean(x) * np.mean(y)

x2 = x * x
m_x2 = np.mean(x2)

mx = np.mean(x)
mx_2 = mx**2

d = m_x2 - mx_2

xy = x * y
m_xy = np.mean(xy)

m_x = np.mean(x)
m_y = np.mean(y)

m_xm_y = m_x * m_y

a = (m_xy - m_xm_y) / d
b = (m_y * m_x2 - m_x * m_xy) / d

_min = np.amin(x)
_max = np.amax(x)

plt.scatter(x, y)
plt.plot([_min, _max], [a * _min + b, a * _max + b])
plt.ticklabel_format(useOffset=False)
plt.show()
