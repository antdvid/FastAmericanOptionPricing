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
        K12 = self.K12(tau, B)
        Phi = self.CDF_pos_dplus(tau, B/self.K)
        return self.PDF_dplus(tau, B/self.K)/(self.sigma * np.sqrt(tau)) + Phi + self.q * (K12)

    def K12(self, tau, B):
        return self.quadrature_sum(self.K12_integrand, tau, B, self.quadrature_num)

    def K12_integrand(self, tau, B_tau, u, B_u):
        if tau - u == 0:
            CDF_dplus = 1
            PDF_dplus = 0
        else:
            dplus = self.dplus(tau - u, B_tau / B_u)
            CDF_dplus = stats.norm.cdf(dplus)
            PDF_dplus = stats.norm.pdf(dplus)
        return np.exp(self.q * u) * CDF_dplus \
            + np.exp(self.q * u)/(self.sigma * np.sqrt(tau - u)) * PDF_dplus

    def K3(self, tau, B):
        return self.quadrature_sum(self.K3_integrand, tau, B, self.quadrature_num)

    def K3_integrand(self, tau, B_tau, u, B_u):
        if tau - u == 0:
            PDF_dminus = 0
        else:
            PDF_dminus = stats.norm.pdf(self.dminus(tau - u, B_tau/ B_u))
        return np.exp(self.r * u)/(self.sigma * np.sqrt(tau - u)) * PDF_dminus

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
        return self.PDF_dplus(tau, Q/self.K)/(self.sigma * np.sqrt(tau) * Q) \
               * (1 - self.dplus(tau, Q/self.K)/(self.sigma * np.sqrt(tau))) \
               + self.q * self.quadrature_sum(self.Dprime_integrand, tau, Q, self.quadrature_num)

    def Dprime_integrand(self, tau, B_tau, u, B_u):
        tau = max(1e-10, tau)
        return np.exp(self.q * u) * self.PDF_dplus(tau-u, B_tau/B_u)/(self.sigma * np.sqrt(tau - u) * B_tau) \
            * (1 - self.dplus(tau - u, B_tau/B_u)/(self.sigma * np.sqrt(tau - u)))
