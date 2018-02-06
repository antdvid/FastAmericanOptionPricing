import numpy as np
import matplotlib.pyplot as plt
import ChebyshevInterpolation as intrp
from random import *


def trial_func(x):
    return np.exp(x)* x


def cheby_to_interval(cheb_points, a, b):
    return a + (b - a) * (cheb_points + 1) * 0.5


def interval_to_cheby(x, a, b):
    return (2*x - b - a) / (b - a)

def perturb(y):
    for i in range(len(y)):
        y[i] += (random() * 2 - 1) * 0.4 * y[i]

#test convergence of chebyshev interpolation
res = []
cheby_point_num = 15
interval = [0, 3]
a = interval[0]
b = interval[1]
x = intrp.ChebyshevInterpolation.get_std_cheby_points(cheby_point_num)
x = cheby_to_interval(x, a, b)
y = trial_func(x)
perturb(y)
cheby_intrp = intrp.ChebyshevInterpolation(y, interval_to_cheby, a, b)
xi = np.linspace(interval[0], interval[1], 100)
yintrp = cheby_intrp.value(xi)
plt.plot(x, y, 'o')
plt.plot(xi, yintrp, '-')
plt.show()

yintrp = cheby_intrp.value(x)
err = np.linalg.norm(np.abs(y - yintrp))
print("err = ", err)
