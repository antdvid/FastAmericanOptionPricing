import numpy as np
import scipy.stats as stats


class EuropeanOption:
    @staticmethod
    def european_option_value(tau, s0, r, q, vol, strike):
        """put option"""
        if tau == 0:
            return max(0, strike - s0)
        d1 = EuropeanOption.d1(tau, s0, r, q, vol, strike)
        d2 = EuropeanOption.d2(tau, s0, r, q, vol, strike)
        return strike * np.exp(-r * tau) * stats.norm.cdf(-d2) - s0 * np.exp(-q * tau) * stats.norm.cdf(-d1)

    @staticmethod
    def european_option_theta(tau, s0, r, q, vol, strike):
        """put option theta"""
        tau = max(tau, 1e-10) # set tau negative
        d1 = EuropeanOption.d1(tau, s0, r, q, vol, strike)
        d2 = EuropeanOption.d2(tau, s0, r, q, vol, strike)
        return r*strike * np.exp(-r * tau) * stats.norm.cdf(-d2) - q * s0 * np.exp(-q * tau)*stats.norm.cdf(-d1) \
            - vol * s0 * np.exp(-q * tau) * stats.norm.pdf(d1)/(2 * np.sqrt(tau))

    @staticmethod
    def d1(tau, s0, r, q, vol, strike):
        return np.log(s0 * np.exp((r-q)*tau)/strike)/(vol * np.sqrt(tau)) + 0.5*vol * np.sqrt(tau)

    @staticmethod
    def d2(tau, s0, r, q, vol, strike):
        return EuropeanOption.d1(tau, s0, r, q, vol, strike) - vol * np.sqrt(tau)