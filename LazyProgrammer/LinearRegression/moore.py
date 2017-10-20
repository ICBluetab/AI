import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

p = re.compile('\d\d\d\d')


def year_to_numeric(year):
    return int(year)

df = pd.read_csv("moore.csv", header=None, sep='\t')
M = df[[2, 1]].as_matrix()

print(M)
