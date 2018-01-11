import numpy as np
import matplotlib.pyplot as plt
import ChebyshevInterpolation as intrp


def test_func(x):
    return np.exp(x)* x

cheby_point_num = 8
cheby_intrp = intrp.ChebyshevInterpolation(cheby_point_num)
x = cheby_intrp.get_std_cheby_points()
y = test_func(x)
xi = np.linspace(-1, 1, 40)
yi = test_func(xi)

yintrp = cheby_intrp.std_cheby_value(xi, y)
plt.plot(xi, yi, '-', xi, yintrp, 'o')
print("error = ", np.linalg.norm(np.abs(yi - yintrp), 2))
plt.show()