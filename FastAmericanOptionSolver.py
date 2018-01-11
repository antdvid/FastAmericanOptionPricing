import numpy as np
import scipy.stats as stats
import scipy.integrate.quadrature
import ChebyshevInterpolation as intrp
import EuropeanOptionSolver as europ


class FastAmericanOptionSolver:
    def __init__(self, riskfree, dividend, volatility, strike, maturity):
        self.r = riskfree
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.collocation_num = 20
        self.iter_tol = 1e-10
        self.shared_B = np.array(self.collocation_num)
        self.shared_tau = np.array(self.collocation_num)
        self.tau_max = self.T

    def solve(self, t, s0):
        tau = self.T - t
        return europ.EuropeanOption.european_option_value(tau, s0, self.r, self.q, self.sigma, self.K)

    def set_collocation_points(self):
        n = self.collocation_num
        z = np.cos(np.arange(self.collocation_num) * np.pi/n)
        x = 0.5 * np.sqrt(self.tau_max) * (1.0 + z)
        np.copyto(self.shared_tau, np.square(x))

    def compute_exercise_boundary(self):
        B = self.get_initial_guess()  # B vector for initial guess
        B_old = 10000 + B

        while np.abs(B_old - B) > self.iter_tol:
            B_old = B.copy()
            B = self.iterate_once(self.shared_tau, B)
            self.shared_B = B
        return B

    def iterate_once(self, tau, B):
        eta = 0.5
        B_new = []
        for tau_i, B_i in tau, B:
            N = self.N_func(tau_i, B_i)
            D = self.D_func(tau_i, B_i)
            Ndot = self.Ndot_func(tau_i, B_i)
            Ddot = self.Ddot_func(tau_i, B_i)
            f = self.K * np.exp(-tau * (self.r - self.q)) * N / D
            fdot = self.K * np.exp(-tau * (self.r - self.q)) * (Ndot / D - Ddot * N / (D * D))
            B_i += eta * (B_i - f) / (fdot - 1)
            B_new.append(B_i)
        return B_new

    def get_initial_guess(self):
        return np.zeros(self.collocation_num, 1)

    def N_func(self, tau, B):
        return self.CDF(self.dminus(tau, B/self.K)) + self.r * self.quadrature(self.N_integrand, 0, tau, tau)

    def D_func(self, tau, B):
        return self.CDF(self.dplus(tau, B/self.K)) + self.q * self.quadrature(self.D_integrand, 0, tau, tau)

    def N_integrand(self, u, tau):
        return np.exp(self.r * u) * self.CDF(self.dminus(tau-u, self.Bfunc(tau)/self.Bfunc(u)))

    def D_integrand(self, u, tau):
        return np.exp(self.q * u) * self.CDF(self.dplus(tau-u, self.Bfunc(tau)/self.Bfunc(u)))

    def Ndot_func(self, tau, Q):
        return self.PDF(self.dminus(tau, Q/self.K))/(self.sigma * np.sqrt(tau) * Q) + self.r * self.quadrature(self.Ndot_integrand, 0, tau, tau)

    def Ddot_func(self, tau, Q):
        return self.PDF(self.dminus(tau, Q/self.K))/(self.sigma * np.sqrt(tau) * Q) + self.r * self.quadrature(self.Ddot_integrand, 0, tau, tau)

    def Ndot_integrand(self, u, tau):
        return np.exp(self.r * u) * self.PDF(self.dminus(tau-u, self.Bfunc(tau)/self.Bfunc(u))) / (
        self.sigma * np.sqrt(tau - u) * self.Bfunc(tau))

    def Ddot_integrand(self, u, tau):
        return np.exp(self.q * u) * self.PDF(self.dplus(tau - u, self.Bfunc(tau) / self.Bfunc(u))) / (
        self.sigma * np.sqrt(tau - u) * self.Bfunc(tau))

    def dminus(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau - 0.5 * self.sigma * self.sigma * tau)/(self.sigma * np.sqrt(tau))

    def dplus(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau + 0.5 * self.sigma * self.sigma * tau)/(self.sigma * np.sqrt(tau))

    def CDF(self, tau):
        return stats.norm.cdf(tau)

    def PDF(self, tau):
        return stats.norm.pdf(tau)

    def Bfunc(self, tau):
        """interpolation on shared_B should be done on Chebyshev points
        with transformation H(sqrt(tau)) = ln(B(tau)/X), X = K min(1, r/q)"""
        return self.chebyshev_func(tau)

    def chebyshev_func(self, tau):
        X = self.K * min(1, self.r / self.q)
        H = np.log(self.shared_B / X)
        cheby_interp = intrp.ChebyshevInterpolation(self.collocation_num)
        to_cheby_point = np.sqrt(4 * tau/self.tau_max) - 1
        q = cheby_interp.std_cheby_single_value(to_cheby_point, H)
        ans = np.exp(np.sqrt(q)) * X
        return ans

    def quadrature(self, func, a, b, tau):
        return scipy.integrate.quadrature(self, func, a, b, tau)
