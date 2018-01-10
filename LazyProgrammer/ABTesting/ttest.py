import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N = 10
df = 2 * N - N  # degrees of freedom
a = np.random.randn(N) + 2
b = np.random.randn(N)
c = np.random.standard_t(df, size=N)

a = np.sort(a)
b = np.sort(b)
c = np.sort(c)

var_a = a.var(ddof=1)
var_b = b.var(ddof=1)

s = np.sqrt( (var_a + var_b)/2)
t = (a.mean() - b.mean()) / (s * np.sqrt(2.0/N))

p = 1 - stats.t.cdf(t, df=df)
print "t:\t", t, "p:\t", 2 * p

t2, p2 = stats.ttest_ind(a, b)
print "t:\t", t2, "p:\t", p2

a_hat = stats.norm.pdf(a, 2, 1)
plt.plot(a, a_hat)

b_hat = stats.norm.pdf(b, 0, 1)
plt.plot(b, b_hat)

c_hat = stats.t.pdf(c, df)
plt.plot(c, c_hat)

plt.scatter(t, stats.t.pdf(t, df))

plt.show()
