from FastAmericanOptionSolverBase import *


class FastAmericanOptionSolverB(FastAmericanOptionSolver):
    def N_func(self, tau, B):
        tau = max(1e-10, tau)
        if self.option_type == qd.OptionType.Put:
            return self.CDF_pos_dminus(tau, B/self.K) \
                + self.r * self.quadrature_sum(self.N_integrand, tau, B, self.quadrature_num)
        else:
            return self.CDF_neg_dminus(tau, B/self.K) \
                + self.r * self.quadrature_sum(self.N_integrand, tau, B, self.quadrature_num)

    def D_func(self, tau, B):
        tau = max(1e-10, tau)
        if self.option_type == qd.OptionType.Put:
            return self.CDF_pos_dplus(tau, B/self.K) + \
                self.q * self.quadrature_sum(self.D_integrand, tau, B, self.quadrature_num)
        else:
            return self.CDF_neg_dplus(tau, B / self.K) + \
                   self.q * self.quadrature_sum(self.D_integrand, tau, B, self.quadrature_num)

    def N_integrand(self, tau, B_tau, u, B_u):
        # every input is a scalar
        if self.option_type == qd.OptionType.Put:
            return np.exp(self.r * u) * self.CDF_pos_dminus(tau-u, B_tau/B_u)
        else:
            return np.exp(self.r * u) * self.CDF_neg_dminus(tau-u, B_tau/B_u)

    def D_integrand(self, tau, B_tau, u, B_u):
        # every input is a scalar
        if self.option_type == qd.OptionType.Put:
            return np.exp(self.q * u) * self.CDF_pos_dplus(tau-u, B_tau/B_u)
        else:
            return np.exp(self.q * u) * self.CDF_neg_dplus(tau - u, B_tau/B_u)

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