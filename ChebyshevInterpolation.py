import numpy as np

class ChebyshevInterpolation:
    def __init__(self, numpoints):
        self.n = numpoints-1

    def get_std_cheby_points(self):
        i = np.arange(0, self.n+1)
        return np.cos(i * np.pi/self.n)

    def std_cheby_value(self, zv, n_y):
        ans = []
        for z in zv:
            ans.append(self.std_cheby_single_value(z, n_y))
        return ans

    def std_cheby_single_value(self, z, n_y):
        """z is the point to be valued between [-1, 1], n_y are the function values at Chebyshev points
        Iteration using Clenshaw algorithm"""
        b0 = self.std_coeff(self.n, n_y) * 0.5
        b1 = 0
        b2 = 0

        for k in range(self.n-1,-1,-1):
            a = self.std_coeff(k, n_y)
            b1, b2 = b0, b1
            b0 = a + 2 * z * b1 - b2
        return 0.5 * (b0 - b2)

    def std_coeff(self, k, n_y):
        ans = 0
        for i in range(0, self.n+1):
            term = n_y[i] * np.cos(i * k * np.pi / self.n)
            if i == 0 or i == self.n:
                term *= 0.5
            ans += term
        ans *= (2.0/self.n)
        return ans