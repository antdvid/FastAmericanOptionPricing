import numpy as np
import scipy.stats as stats
import ChebyshevInterpolation as intrp
import EuropeanOptionSolver as europ
import QDplusAmericanOptionSolver as qd
import numpy.linalg as alg
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as plt


class FastAmericanOptionSolver:
    def __init__(self, riskfree, dividend, volatility, strike, maturity):
        self.r = riskfree
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.collocation_num = 12
        self.quadrature_num = 24
        self.integration_num = 2 * self.quadrature_num
        self.iter_tol = 1e-5
        self.shared_B0 = []
        self.shared_B = []
        self.shared_B_old = []
        self.shared_tau = []
        self.tau_max = self.T
        self.european_put_price = 0

        # points and weights for Guassian integration
        self.y = [-0.90618, -0.538469, 0, 0.538469, 0.90618]
        self.w = [0.236927, 0.478629,  0, 0.478629, 0.236927]
        self.shared_Bu = [None] * len(self.y)
        self.shared_u = [None] * len(self.y)
        self.tau_cache = -1
        self.integration_num_cache = -1

        self.iter_records = []

        # Debug switch
        self.DEBUG = True
        self.use_derivative = False

    def solve(self, t, s0):
        tau = self.T - t
        self.set_collocation_points()
        ####check collocation points are done###
        self.debug("step 1. checking collocation points ...")
        self.debug("collocation point = {0}".format(self.shared_tau))
        ########################################

        ####check numerical integration are correct###
        self.debug("step 3. checking numerical integration ...")
        self.test_numerical_integration()
        ########################################

        self.compute_exercise_boundary()

        ##### check exercise boundary ###########
        self.debug("step 6. checking exercise boundary ...")
        self.debug("exercise boundary = {0}".format(self.shared_B))
        self.debug("match condition err = {0}".format(self.check_value_match_condition2()))
        ########################################

        v = self.american_put_with_known_boundary(tau, s0, self.r, self.q, self.sigma, self.K)
        return v

    def test_numerical_integration(self):
        self.set_initial_guess()
        tau = 3.0
        s0 = 2
        analy_res = s0 * 0.5 * (np.exp(tau * tau) - 1)
        num_res = self.quadrature_sum(self.test_integrand, tau, s0, self.quadrature_num)
        self.debug("analytical sol = {0}, numerical sol = {1}, err = {2}".format(analy_res, num_res, abs(analy_res - num_res)))

    def test_integrand(self, tau, S, u, Bu):
        return S * u * np.exp(u * u)

    def american_put_with_known_boundary(self, tau, s0, r, q, sigma, K):
        v = europ.EuropeanOption.european_option_value(tau, s0, r, q, sigma, K)
        self.european_put_price = v  # save european price
        v += self.quadrature_sum(self.v_integrand_1, tau, s0, self.integration_num)
        v -= self.quadrature_sum(self.v_integrand_2, tau, s0, self.integration_num)
        return v

    def compute_exercise_boundary(self):
        self.set_initial_guess()
        ##################################
        self.debug("step 4. checking QD+ alogrithm ...")
        self.debug("B guess = {0}".format(self.shared_B))
        self.debug("tau  = {0}".format(self.shared_tau))
        ##################################

        ##################################
        self.debug("step 5. starting iteration ...")
        ##################################
        iter_count = 0
        iter_err = 1
        while iter_err > self.iter_tol:
            iter_count += 1
            B_old = self.shared_B.copy()
            self.shared_B = self.iterate_once(self.shared_tau, B_old)
            self.shared_B_old = B_old
            iter_err = self.norm1_error(B_old, self.shared_B)
            self.debug("  iter = {0}, err = {1}".format(iter_count, self.norm1_error(B_old, self.shared_B)))
            #self.debug("match condition err1 = {0}".format(self.check_value_match_condition()))
            #self.debug("match condition err2 = {0}".format(self.check_value_match_condition2()))
            self.iter_records.append((iter_count, self.check_value_match_condition2()))

    def iterate_once(self, tau, B):
        """the for-loop can be parallelized"""
        eta = 1.0
        B_new = []
        #self.check_f_with_B()
        for i in range(len(tau)):
            tau_i = tau[i]
            B_i = B[i]
            f_and_fprime = self.compute_f_and_fprime(tau_i, B_i)
            f = f_and_fprime[0]
            # if len(self.shared_B_old) != 0:
            #     num_fprime = self.compute_fprime_numerical(tau_i, B_i, self.shared_B_old[i])
            # else:
            #     num_fprime = f_and_fprime[1]

            ####
            if self.use_derivative:
                fprime = f_and_fprime[1]
            else:
                fprime = 0.0

            ###
            # print("tau_i = ", tau_i, "analy fprime = ", f_and_fprime[1], "numr fprime = ", num_fprime)
            if tau_i == 0:
                B_i = min(self.K, self.K * self.r / self.q)
            else:
                B_i += eta * (B_i - f)/(fprime - 1)

            B_new.append(B_i)
        return B_new

    def compute_integration_terms(self, tau, num_points):
        """compute u between 0, tau_i"""
        if tau == self.tau_cache and num_points == self.integration_num_cache:
            return
        else:
            self.tau_cache = tau
            self.integration_num_cache = num_points

        points_weights = legendre.leggauss(num_points)
        self.y = points_weights[0]
        self.w = points_weights[1]
        self.shared_Bu = [None] * len(self.y)
        self.shared_u = [None] * len(self.y)

        N = len(self.y)
        for i in range(N):
            self.shared_u[i] = tau - tau * np.square(1 + self.y[i])/4.0
            self.shared_Bu[i] = self.chebyshev_func(self.shared_u[i])

    def v_integrand_1(self, tau, S, u, Bu):
        # every input is scalar
        return self.r * self.K * np.exp(-self.r * (tau - u)) * self.CDF_neg_dminus(tau-u, S/Bu)

    def v_integrand_2(self, tau, S, u, Bu):
        # every input is scalar
        return self.q * S * np.exp(-self.q * (tau - u)) * self.CDF_neg_dplus(tau-u, S/Bu)

    def set_collocation_points(self):
        cheby_intrp = intrp.ChebyshevInterpolation(self.collocation_num)
        self.shared_tau = self.to_orig_point(cheby_intrp.get_std_cheby_points(), self.tau_max)

    def debug(self, message):
        if self.DEBUG == True:
            print(message)
            print("")

    def norm1_error(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        return alg.norm(np.abs(x1 - x2))

    def to_cheby_point(self, x, x_max):
        return np.sqrt(4 * x / x_max) - 1

    def to_orig_point(self, c, x_max):
        return np.square(c + 1) * x_max / 4

    def jac(self, a, b, x):
        """this function defines transformation jacobian for y = f(x): dy = jac * dx"""
        return 0.5 * (b - a) * (1 + x)

    def set_initial_guess(self):
        """get initial guess for all tau_i using QD+ algorithm"""
        qd_solver = qd.QDplus(self.r, self.q, self.sigma, self.K)
        res = []
        for tau_i in self.shared_tau:
            res.append(qd_solver.compute_exercise_boundary(tau_i))
        self.shared_B = res
        self.shared_B0 = res.copy()

    def compute_f_and_fprime(self, tau_i, B_i):
        if tau_i == 0:
            return [self.K, 0]
        N = self.N_func(tau_i, B_i)
        D = self.D_func(tau_i, B_i)
        Ndot = self.Nprime_func(tau_i, B_i)
        Ddot = self.Dprime_func(tau_i, B_i)
        f = self.K * np.exp(-tau_i * (self.r - self.q)) * N / D
        fprime = self.K * np.exp(-tau_i * (self.r - self.q)) * (Ndot / D - Ddot * N / (D * D))
        return [f, fprime]

    def compute_fprime_numerical(self, tau_i, B_i, B_i_old):
        if B_i == B_i_old:
            return 0
        up_res = self.compute_f_and_fprime(tau_i, B_i)
        down_res = self.compute_f_and_fprime(tau_i, B_i_old)
        f_up = up_res[0]
        f_down = down_res[0]
        return (f_up - f_down)/(B_i - B_i_old)

    def N_func(self, tau, B):
        tau = max(1e-10, tau)
        return self.CDF_pos_dminus(tau, B/self.K) \
               + self.r * self.quadrature_sum(self.N_integrand, tau, B, self.quadrature_num)

    def D_func(self, tau, B):
        tau = max(1e-10, tau)
        return self.CDF_pos_dplus(tau, B/self.K) + \
               self.q * self.quadrature_sum(self.D_integrand, tau, B, self.quadrature_num)

    def N_integrand(self, tau, B_tau, u, B_u):
        # every input is a scalar
        return np.exp(self.r * u) * self.CDF_pos_dminus(tau-u, B_tau/B_u)

    def D_integrand(self, tau, B_tau, u, B_u):
        # every input is a scalar
        return np.exp(self.q * u) * self.CDF_pos_dplus(tau-u, B_tau/B_u)

    def Nprime_func(self, tau, Q):
        tau = max(1e-10, tau)
        return self.PDF_dminus(tau, Q/self.K)/(self.sigma * np.sqrt(tau) * Q) \
               + self.r * self.quadrature_sum(self.Nprime_integrand, tau, Q, self.quadrature_num)

    def Dprime_func(self, tau, Q):
        tau = max(1e-10, tau)
        return self.PDF_dplus(tau, Q/self.K)/(self.sigma * np.sqrt(tau) * Q) \
               + self.q * self.quadrature_sum(self.Dprime_integrand, tau, Q, self.quadrature_num)

    def Nprime_integrand(self, tau, B_tau, u, B_u):
        tau = max(1e-10, tau)
        return np.exp(self.r * u) * self.PDF_dminus(tau-u, B_tau/B_u) \
               / (self.sigma * np.sqrt(tau - u) * B_tau)

    def Dprime_integrand(self, tau, B_tau, u, B_u):
        tau = max(1e-10, tau)
        return np.exp(self.q * u) * self.PDF_dplus(tau - u, B_tau / B_u) \
               / (self.sigma * np.sqrt(tau - u) * B_tau)

    def dminus(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau - 0.5 * self.sigma * self.sigma * tau)/(self.sigma * np.sqrt(tau))

    def dplus(self, tau, z):
        return (np.log(z) + (self.r - self.q)*tau + 0.5 * self.sigma * self.sigma * tau)/(self.sigma * np.sqrt(tau))

    def CDF_neg_dminus(self, tau, z):
        # phi(-d-)
        if tau == 0:
            return 0
        else:
            return stats.norm.cdf(-self.dminus(tau, z))

    def CDF_pos_dminus(self, tau, z):
        # phi(+d-)
        if tau == 0:
            return 1
        else:
            return stats.norm.cdf(self.dminus(tau, z))

    def CDF_neg_dplus(self, tau, z):
        # phi(-d+)
        if tau == 0:
            return 0
        else:
            return stats.norm.cdf(-self.dplus(tau, z))

    def CDF_pos_dplus(self, tau, z):
        # phi(+d+)
        if tau == 0:
            return 1
        else:
            return stats.norm.cdf(self.dplus(tau, z))

    def PDF_dminus(self, tau, z):
        if tau == 0:
            return 0
        else:
            return stats.norm.pdf(self.dminus(tau, z))

    def PDF_dplus(self, tau, z):
        if tau == 0:
            return 0
        else:
            return stats.norm.pdf(self.dplus(tau, z))

    def chebyshev_func(self, tau):
        cheby_interp = intrp.ChebyshevInterpolation(self.collocation_num)
        to_cheby_point = self.to_cheby_point(tau, self.tau_max)
        X = self.K * min(1, self.r/self.q)
        #this transformation significantly reduces the number of iterations
        H = np.square(np.log(np.array(self.shared_B) / X))
        ans = cheby_interp.std_cheby_value([to_cheby_point], H)[0]
        ans = np.exp(-np.sqrt(max(0, ans))) * X
        return ans

    def quadrature_sum(self, integrand, tau, S, num_points):
        # tau, S are scalar, u and Bu are vectors for integration
        # u, Bu and y, w should have the same number of points

        # important, recalcualte integration points and weights
        self.compute_integration_terms(tau, num_points)
        u = self.shared_u
        Bu = self.shared_Bu

        assert len(u) == len(Bu) and len(u) == len(self.w)
        if tau == 0:
            return 0
        ans = 0
        for i in range(len(u)):
            adding = integrand(tau, S, u[i], Bu[i]) * self.w[i] * self.jac(0, tau, self.y[i])
            ans += adding
        return ans

    def check_value_match_condition(self):
        left = []
        right = []
        for tau_i, B_i in zip(self.shared_tau, self.shared_B):
            left.append(self.K - B_i)
            right.append(self.american_put_with_known_boundary(tau_i, B_i, self.r, self.q, self.sigma, self.K))
        return self.norm1_error(left, right)

    def check_value_match_condition2(self):
        left = []
        right = []
        for tau_i, B_i in zip(self.shared_tau, self.shared_B):
            left.append(self.N_func(tau_i, B_i) * self.K * np.exp(-self.r * tau_i))
            right.append(self.D_func(tau_i, B_i) * B_i * np.exp(- self.q * tau_i))
        return self.norm1_error(left, right)

    def check_f_with_B(self):
        N = 20
        tau = 0.5
        B = np.linspace(50, 100, N)
        fprime = []
        f = []
        for Bi in B:
            res = self.compute_f_and_fprime(tau, Bi)
            fprime.append(res[1])
            f.append(res[0])

        print(fprime)

        plt.subplot(1,2,1)
        plt.plot(B, f, 'o-')
        plt.xlabel("B")
        plt.ylabel("f")
        plt.subplot(1,2,2)
        plt.plot(B, fprime, 'o-r')
        plt.xlabel("B")
        plt.ylabel("f prime")
        plt.show()

        exit()