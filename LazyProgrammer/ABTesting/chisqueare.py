import numpy as np
import matplotlib.pyplot as plt
from sipy.stats import chi2

class DataGenerator:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def next(self):
        click1 = 1 if (np.random.random() < slef.p1) else 0
        click2 = 1 if (np.random.random() < slef.p2) else 0
