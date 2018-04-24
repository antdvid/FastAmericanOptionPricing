import numpy as np
import matplotlib.pyplot as plt
import ChebyshevInterpolation as intrp


def test_func(x):
    return np.exp(x)* x
    #return np.cos(x) * 100

#test convergence of chebyshev interpolation
res = []
for cheby_point_num in range(21):
    if cheby_point_num < 2:
        continue
    x = intrp.ChebyshevInterpolation.get_std_cheby_points(cheby_point_num)
    y = test_func(x)
    cheby_intrp = intrp.ChebyshevInterpolation(y, lambda t, a, b: t, -1, 1)
    xi = np.linspace(-1, 1, 50)
    yi = test_func(xi)
    yintrp = cheby_intrp.value(xi)
    err = np.linalg.norm(np.abs(yi - yintrp))
    print(cheby_point_num, err)
    res.append((cheby_point_num, err))

pts = [float(e[0]) for e in res]
errs = [e[1] for e in res]
plt.xticks(pts)
plt.loglog(pts, errs, '-o')
plt.loglog(pts, np.power(pts, -2), '--r')
plt.loglog(pts, np.power(pts, -3), '--g')
plt.loglog(pts, np.power(pts, -4), '--b')
plt.legend(['chebyshev', '2nd order', '3rd order', '4th order'])
plt.xlabel('Number of points')
plt.ylabel('Norm-1 error')
plt.show()