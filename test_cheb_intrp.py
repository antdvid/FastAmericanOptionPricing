import numpy as np
import matplotlib.pyplot as plt
import ChebyshevInterpolation as intrp
from random import *


def test_func(x):
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
cheby_intrp = intrp.ChebyshevInterpolation(cheby_point_num)
x = cheby_intrp.get_std_cheby_points()
x = cheby_to_interval(x, a, b)
y = test_func(x)
perturb(y)
xi = np.linspace(interval[0], interval[1], 100)
#yi = test_func(xi)
yintrp = cheby_intrp.std_cheby_value(interval_to_cheby(xi, a, b), y)
plt.plot(x, y, 'o')
plt.plot(xi, yintrp, '-')
plt.show()

yintrp = cheby_intrp.std_cheby_value(interval_to_cheby(x, a, b), y)
err = np.linalg.norm(np.abs(y - yintrp))
print("err = ", err)
