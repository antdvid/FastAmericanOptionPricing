from FastAmericanOptionSolverB import *


class FastAmericanOptionSolverA(FastAmericanOptionSolver):
    def N_func(self, tau, B):
        if tau == 0:
            return 0
        K3 = self.K3(tau, B)
        return self.PDF_dminus(tau, B/self.K)/(self.sigma * np.sqrt(tau)) + self.r * K3

    def D_func(self, tau, B):
        if tau == 0:
            return 0
        K1 = self.K1(tau, B)
        K2 = self.K2(tau, B)
        Phi = self.CDF_pos_dplus(tau, B/self.K)
        return self.PDF_dplus(tau, B/self.K)/(self.sigma * np.sqrt(tau)) + Phi + self.q * (K1 + K2)

    def K1(self, tau, B):
        return self.quadrature_sum(self.K1_integrand, tau, B, self.quadrature_num)

    def K1_integrand(self, tau, B_tau, u, B_u):
        return np.exp(self.q * u) * self.CDF_pos_dplus(tau - u, B_tau/B_u)

    def K2(self, tau, B):
        return self.quadrature_sum(self.K2_integrand, tau, B, self.quadrature_num)

    def K2_integrand(self, tau, B_tau, u, B_u):
        return np.exp(self.q * u)/(self.sigma * np.sqrt(tau - u)) * self.PDF_dplus(tau - u, B_tau / B_u)

    def K3(self, tau, B):
        return self.quadrature_sum(self.K3_integrand, tau, B, self.quadrature_num)

    def K3_integrand(self, tau, B_tau, u, B_u):
        return np.exp(self.r * u)/(self.sigma * np.sqrt(tau - u)) * self.PDF_dminus(tau - u, B_tau/B_u)

    def Nprime_func(self, tau, Q):
        tau = max(1e-10, tau)
        return -self.dminus(tau, Q/self.K) * self.PDF_dminus(tau, Q/self.K) / (Q * self.sigma * self.sigma * tau) \
               - self.r * self.quadrature_sum(self.Nprime_integrand, tau, Q, self.quadrature_num)

    def Nprime_integrand(self, tau, B_tau, u, B_u):
        tau = max(1e-10, tau)
        return np.exp(self.r * u) * self.dminus(tau-u, B_tau/B_u) / (B_tau * self.sigma * self.sigma * (tau - u))\
               * self.PDF_dminus(tau-u, B_tau/B_u)

    def Dprime_func(self, tau, Q):
        tau = max(1e-10, tau)
        #K_star = self.K * np.exp(-(self.r - self.q) * tau)
        #return -K_star/Q * self.dminus(tau, Q/self.K)*self.PDF_dminus(tau, Q/self.K)/(Q * self.sigma * self.sigma * tau) \
        #       - self.q * K_star/Q * self.quadrature_sum(self.Dprime_integrand, tau, Q, self.quadrature_num)
        return -self.dplus(tau, Q/self.K) * self.PDF_dplus(tau, Q/self.K) /(self.sigma * self.sigma * tau * Q) \
            + self.PDF_dplus(tau, Q/self.K)/(self.sigma * Q * np.sqrt(tau))\
            + self.q * self.quadrature_sum(self.Dprime_integrand, tau, Q, self.quadrature_num)

    def Dprime_integrand(self, tau, B_tau, u, B_u):
        tau = max(1e-10, tau)
        #return B_u/self.K * np.exp(self.r * u) * self.dminus(tau-u, B_tau/B_u)/(self.sigma * self.sigma * (tau - u)) \
        #       * self.PDF_dminus(tau-u, B_tau/B_u)
        return np.exp(self.q * u) * self.PDF_dplus(tau-u, B_tau/B_u)/(self.sigma * np.sqrt(tau - u) * B_tau) \
            - (np.exp(self.q * u) * self.dplus(tau-u, B_tau/B_u) * self.PDF_dplus(tau-u, B_tau/B_u))\
              /(self.sigma * self.sigma * (tau - u) * B_tau)
