import numpy as np

class ChebyshevInterpolation:
    def __init__(self, ynodes, x_to_cheby, x_min, x_max):
        numpoints = len(ynodes)
        self.n = numpoints-1
        self.a = [0] * numpoints
        self.x_to_cheby = x_to_cheby
        self.x_min = x_min
        self.x_max = x_max
        for k in range(numpoints):
            self.a[k] = self.std_coeff(k, ynodes)

    @staticmethod
    def get_std_cheby_points(numpoints):
        i = np.arange(0, numpoints+1)
        return np.cos(i * np.pi/numpoints)

    def value(self, zv):
        ans = []
        to_cheby = zv
        if self.x_to_cheby is not None:
            to_cheby = self.x_to_cheby(zv, self.x_min, self.x_max)
        for z in to_cheby:
            ans.append(self.std_cheby_single_value(z))
        return ans

    def std_cheby_single_value(self, z):
        """z is the point to be valued between [-1, 1], n_y are the function values at Chebyshev points
        Iteration using Clenshaw algorithm"""
        b0 = self.a[self.n] * 0.5
        b1 = 0
        b2 = 0

        for k in range(self.n - 1, -1, -1):
            b1, b2 = b0, b1
            b0 = self.a[k] + 2 * z * b1 - b2
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