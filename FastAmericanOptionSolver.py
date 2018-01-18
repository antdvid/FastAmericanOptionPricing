import numpy as np
import scipy.stats as stats
import ChebyshevInterpolation as intrp
import EuropeanOptionSolver as europ
import QDplusAmericanOptionSolver as qd
import numpy.linalg as alg
import math
import matplotlib.pyplot as plt


class FastAmericanOptionSolver:
    def __init__(self, riskfree, dividend, volatility, strike, maturity):
        self.r = riskfree
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.collocation_num = 15
        self.iter_tol = 1e-5
        self.shared_B = []
        self.shared_tau = []
        self.tau_max = self.T

        # points and weights for Guassian integration
        self.y = [-0.90618, -0.538469, 0, 0.538469, 0.90618]
        self.w = [0.236927, 0.478629,  0, 0.478629, 0.236927]
        self.shared_Bu = [None] * len(self.y)
        self.shared_u = [None] * len(self.y)

    def solve(self, t, s0):
        tau = self.T - t
        self.set_collocation_points()
        ####check collocation points are done###
        self.debug("step 1. checking collocation points ...")
        self.debug("collocation point = {0}".format(self.shared_tau))
        ########################################

        self.compute_exercise_boundary()

        ##### check exercise boundary ###########
        self.debug("step 4. checking exercise boundary ...")
        self.debug("exercise boundary = {0}".format(self.shared_B))
        ########################################

        v = europ.EuropeanOption.european_option_value(tau, s0, self.r, self.q, self.sigma, self.K)
        v += self.quadrature_sum(self.v_integrand_1, tau, s0, self.shared_u, self.shared_Bu)
        v += self.quadrature_sum(self.v_integrand_2, tau, s0, self.shared_u, self.shared_Bu)
        return v

    def v_integrand_1(self, tau, S, u, Bu):
        # every input is scalar
        dminus = self.dminus(tau-u, S/Bu)
        return self.r * self.K * np.exp(-self.r * (tau - u)) * stats.norm.cdf(-dminus)

    def v_integrand_2(self, tau, S, u, Bu):
        # every input is scalar
        dplus = self.dplus(tau-u, S/Bu)
        return self.q * S * np.exp(-self.q * (tau - u)) * stats.norm.cdf(-dplus)

    def set_collocation_points(self):
        n = self.collocation_num
        z = np.cos(np.arange(self.collocation_num) * np.pi/n)
        x = 0.5 * np.sqrt(self.tau_max) * (1.0 + z)
        self.shared_tau = np.square(x)

    def compute_exercise_boundary(self):
        self.set_initial_guess()
        ##################################
        self.debug("step 2. checking QD+ alogrithm ...")
        self.debug("B guess = {0}".format(self.shared_B))
        ##################################

        ##################################
        self.debug("step 3. starting iteration ...")
        ##################################
        iter_count = 0
        iter_err = 1
        while iter_err > self.iter_tol:
            iter_count += 1
            B_old = self.shared_B.copy()
            self.shared_B = self.iterate_once(self.shared_tau, B_old)
            iter_err = self.norm2(B_old, self.shared_B)
            self.debug("  iter = {0}, err = {1}".format(iter_count, self.norm2(B_old, self.shared_B)))

    def debug(self, message):
        DEBUG = True
        if DEBUG == True:
            print(message)
            print("")

    def norm2(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return alg.norm(np.abs(x1 - x2))

    def iterate_once(self, tau, B):
        """the for-loop can be parallelized"""
        eta = 0.8
        B_new = []
        for tau_i, B_i in zip(tau, B):
            #compute u and Bu for integration
            self.compute_integration_terms(tau_i)

            N = self.N_func(tau_i, B_i)
            D = self.D_func(tau_i, B_i)
            Ndot = self.Nprime_func(tau_i, B_i)
            Ddot = self.Dprime_func(tau_i, B_i)
            f = self.K * np.exp(-tau_i * (self.r - self.q)) * N / D
            fdot = 0
            #fdot = self.K * np.exp(-tau_i * (self.r - self.q)) * (Ndot / D - Ddot * N / (D * D))
            B_i += eta * (B_i - f) / (fdot - 1)
            B_new.append(B_i)
        return B_new

    def compute_integration_terms(self, tau):
        """compute u between 0, tau_i"""
        N = len(self.y)
        for i in range(N):
            self.shared_u[i] = tau - tau * np.square(1 + self.y[i])/4.0
            self.shared_Bu[i] = self.chebyshev_func(self.shared_u[i])

        for i in range(N):
            self.check_chebyshev_intrp(self.shared_Bu[i],self.shared_u[i])
        #print("u = ", self.shared_u)
        #print("Bu = ", self.shared_Bu)

    def check_chebyshev_intrp(self, Bu_i, tau_i):
        if not math.isnan(Bu_i):
            return
        print("corrupt! B is nan")
        H = self.shared_B
        cheby_interp = intrp.ChebyshevInterpolation(self.collocation_num)
        tau = np.linspace(0, self.tau_max, 50)
        #add checking point
        np.append(tau, tau_i)
        np.sort(tau)
        #
        to_cheby_point = np.sqrt(4 * tau / self.tau_max) - 1
        q = cheby_interp.std_cheby_value(to_cheby_point, H)
        ans = q
        plt.plot(tau, ans, 'o-')
        plt.plot([tau_i, tau_i], [0, 100], 'r--')
        plt.plot(self.shared_tau, self.shared_B, 'or')
        plt.show()
        exit()

    def set_initial_guess(self):
        """get initial guess for all tau_i using QD+ algorithm"""
        qd_solver = qd.QDplus(self.r, self.q, self.sigma, self.K)
        res = []
        for tau_i in self.shared_tau:
            res.append(qd_solver.compute_exercise_boundary(tau_i))
        self.shared_B = res

    def N_func(self, tau, B):
        return self.CDF(self.dminus(tau, B/self.K)) \
               + self.r * self.quadrature_sum(self.N_integrand, tau, B, self.shared_u, self.shared_Bu)

    def D_func(self, tau, B):
        return self.CDF(self.dplus(tau, B/self.K)) + \
               self.q * self.quadrature_sum(self.D_integrand, tau, B, self.shared_u, self.shared_Bu)

    def N_integrand(self, tau, B_tau, u, B_u):
        # every input is a scalar
        return np.exp(self.r * u) * self.CDF(self.dminus(tau-u, B_tau/B_u))

    def D_integrand(self, tau, B_tau, u, B_u):
        # every input is a scalar
        return np.exp(self.q * u) * self.CDF(self.dplus(tau-u, B_tau/B_u))

    def Nprime_func(self, tau, Q):
        return self.PDF(self.dminus(tau, Q/self.K))/(self.sigma * np.sqrt(tau) * Q) \
               + self.r * self.quadrature_sum(self.Nprime_integrand, tau, Q, self.shared_u, self.shared_Bu)

    def Dprime_func(self, tau, Q):
        return self.PDF(self.dminus(tau, Q/self.K))/(self.sigma * np.sqrt(tau) * Q) \
               + self.r * self.quadrature_sum(self.Dprime_integrand, tau, Q, self.shared_u, self.shared_Bu)

    def Nprime_integrand(self, tau, B_tau, u, B_u):
        return 0
        #return np.exp(self.r * u) * self.PDF(self.dminus(tau-u, B_tau/B_u)) \
         #      / (self.sigma * np.sqrt(tau - u) * B_tau)

    def Dprime_integrand(self, tau, B_tau, u, B_u):
        return 0
        #return np.exp(self.q * u) * self.PDF(self.dplus(tau - u, B_tau / B_u)) \
         #      / (self.sigma * np.sqrt(tau - u) * B_tau)

    def dminus(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau - 0.5 * self.sigma * self.sigma * tau)/(self.sigma * np.sqrt(tau))

    def dplus(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau + 0.5 * self.sigma * self.sigma * tau)/(self.sigma * np.sqrt(tau))

    def CDF(self, tau):
        return stats.norm.cdf(tau)

    def PDF(self, tau):
        return stats.norm.pdf(tau)

    def chebyshev_func(self, tau):
        cheby_interp = intrp.ChebyshevInterpolation(self.collocation_num)
        to_cheby_point = np.sqrt(4 * tau / self.tau_max) - 1
        ans = cheby_interp.std_cheby_value([to_cheby_point], self.shared_B)[0]
        return ans

    def quadrature_sum(self, integrand, tau, S, u, Bu):
        # tau, S are scalar, u and Bu are vectors for integration
        # u, Bu and y, w should have the same number of points
        assert len(u) == len(Bu) and len(u) == len(self.w)
        ans = 0
        for i in range(len(u)):
            adding =  integrand(tau, S, u[i], Bu[i]) * self.w[i]
            ans += adding
        return ans


