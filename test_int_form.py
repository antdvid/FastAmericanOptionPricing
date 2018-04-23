import numpy as np
import ChebyshevInterpolation as intrp
import numpy.polynomial.legendre as legendre


def to_orig_point(c, x_min, x_max):
    return np.square(c + 1) * (x_max - x_min) / 4 + x_min

collocation_num = 12
num_points = 24
tau_min = 0
tau_max = 3
tau = 8e-4
cheby_points = intrp.ChebyshevInterpolation.get_std_cheby_points(collocation_num)
tau_v = to_orig_point(cheby_points, tau_min, tau_max)
points_weights = legendre.leggauss(num_points)
y = points_weights[0]
w = points_weights[1]
u = tau - tau * np.square(1 + y)/4.0

print("c = ", cheby_points)
print("tau = ", tau_v)
print("y = ", y)
print("u = ", u)

