import numpy as np
import scipy.stats as stats
import scipy.optimize
import EuropeanOptionSolver as europ

class QDplus:
    """QD+ alogrithm for computing approximated american option price"""
    def __init__(self, riskfree, dividend, volatility, strike):
        self.r = riskfree
        self.q = dividend
        self.sigma = volatility
        self.K = strike

        # miscellaneous with tau only
        self.v_M = 0
        self.v_N = 0
        self.v_h = 0
        self.v_qQD = 0
        self.v_qQDdot = 0

        # miscellaneous terms with tau and S
        self.v_p = 0
        self.v_theta = 0
        self.v_c = 0
        self.v_b = 0
        self.v_d1 = 0
        self.v_d2 = 0
        self.v_dlogSdh = 0

        self.exercise_boundary = 0

    def price(self, tau, S):
        self.exercise_boundary = Sb = self.compute_exercise_boundary(tau)
        err = self.exercise_boundary_func(Sb, tau)
        print("err = ", err)

        self.compute_miscellaneous(tau, Sb)
        qQD = self.v_qQD
        c = self.v_c
        b = self.v_b
        pS = europ.EuropeanOption.european_option_value(tau, S, self.r, self.q, self.sigma, self.K)
        pSb = europ.EuropeanOption.european_option_value(tau, Sb, self.r, self.q, self.sigma, self.K)

        print("Sb = ", Sb, ", pS = ", pS, ", pSb = ", pSb)
        return pS + (self.K - Sb - pSb)/(1 - b * np.square(np.log(S/Sb)) - c * np.log(S/Sb)) * np.power(S/Sb, qQD)

    def compute_exercise_boundary(self, tau):
        x0 = np.ndarray(1)
        x0[0] = self.K
        #return scipy.optimize.brentq(self.exercise_boundary_func, a=1e-5, b=2*self.K, args=tau)
        return scipy.optimize.root(self.exercise_boundary_func,x0=self.K, args=(tau,)).x[0]

    def compute_miscellaneous(self, tau, S):
        #order cannot be changed
        self.v_N = self.N()
        self.v_M = self.M()
        self.v_h = self.h(tau)
        self.v_qQD = self.q_QD()
        self.v_qQDdot = self.q_QD_dot()
        self.v_d1 = europ.EuropeanOption.d1(tau, S, self.r, self.q, self.sigma, self.K)
        self.v_d2 = europ.EuropeanOption.d2(tau, S, self.r, self.q, self.sigma, self.K)
        self.v_p = europ.EuropeanOption.european_option_value(tau, S, self.r, self.q, self.sigma, self.K)
        self.v_theta = europ.EuropeanOption.european_option_theta(tau, S, self.r, self.q, self.sigma, self.K)
        self.v_dlogSdh = self.dlogSdh(tau, S)
        self.v_c = self.c(tau, S)
        self.v_b = self.b(tau, S)

    def exercise_boundary_func(self, S, tau):
        self.compute_miscellaneous(tau, S)
        qQD = self.v_qQD
        p = self.v_p
        c = self.v_c
        d1 = self.v_d1
        ans = (1 - np.exp(-self.q * tau) * stats.norm.cdf(-d1)) * S + (qQD + c) * (self.K - S - p)
        return ans

    def q_QD(self):
        N = self.v_N
        M = self.v_M
        h = self.v_h
        return -0.5*(N-1) - 0.5 * np.sqrt((N-1)*(N-1) + 4 * M/h)

    def q_QD_dot(self):
        N = self.v_N
        M = self.v_M
        h = self.v_h
        return M/(h * h * np.sqrt((N-1)*(N-1) + 4*M/h))

    def c0(self, tau, S):
        N = self.v_N
        M = self.v_M
        h = self.v_h
        qQD = self.v_qQD
        qQDdot = self.v_qQDdot
        p = self.v_p
        theta = self.v_theta
        c = self.v_c
        d1 = self.v_d1
        d2 = self.v_d2
        return - (1-h)*M/(2*qQD + N - 1) * (1/h - (theta*np.exp(self.r * tau))/(self.r*(self.K - S - p)) + qQDdot/(2*qQD+N-1))

    def c(self, tau, S):
        r = self.r
        q = self.q
        N = self.v_N
        M = self.v_M
        h = self.v_h
        qQD = self.v_qQD
        qQDdot = self.v_qQDdot
        p = self.v_p
        theta = self.v_theta
        c = self.v_c
        d1 = self.v_d1
        d2 = self.v_d2
        dlogSdh = self.v_dlogSdh
        c0 = self.c0(tau, S)

        return c0 - ((1-h)*M)/(2*qQD + N - 1) \
            * ((1 - np.exp(-q * tau)*stats.norm.cdf(-d1))/(self.K - S - p) + qQD/S)\
            * dlogSdh

    def b(self, tau, S):
        N = self.v_N
        M = self.v_M
        h = self.v_h
        qQD = self.v_qQD
        qQDdot = self.v_qQDdot
        p = self.v_p
        theta = self.v_theta
        c = self.v_c
        d1 = self.v_d1
        d2 = self.v_d2
        return ((1-h)*M*qQDdot)/(2*(2*qQD + N - 1))

    def dlogSdh(self, tau, S):
        N = self.v_N
        M = self.v_M
        h = self.v_h
        qQD = self.v_qQD
        qQDdot = self.v_qQDdot
        p = self.v_p
        theta = self.v_theta
        c = self.v_c
        d1 = self.v_d1
        d2 = self.v_d2
        r = self.r
        q = self.q

        dFdh = qQD * theta * np.exp(self.r * tau)/self.r + qQDdot * (self.K - S - p) \
            + (S * self.q *np.exp(-self.q*tau) * stats.norm.cdf(-d1))/(r * (1-h)) \
            - (S * np.exp(-self.q * tau) * stats.norm.pdf(d1))/(2*r*tau*(1-h))\
            * (2*np.log(S/self.K)/(self.sigma * np.sqrt(tau)) - d1)

        dFdS = (1 - qQD) * (1 - np.exp(-q * tau) * stats.norm.cdf(-d1)) \
                + (np.exp(-q * tau) * stats.norm.pdf(d1))/(self.sigma * np.sqrt(tau))

        return -dFdh/dFdS

    def h(self, tau):
        return 1 - np.exp(-self.r * tau)

    def M(self):
        return 2 * self.r / (self.sigma * self.sigma)

    def N(self):
        return 2 * (self.r - self.q) / (self.sigma * self.sigma)
