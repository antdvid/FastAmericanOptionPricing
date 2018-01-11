import numpy as np
import scipy.stats as stats


class EuropeanOption:
    @staticmethod
    def european_option_value(tau, s0, r, q, vol, strike):
            if tau < 1e-5:
                return max(0, s0 - strike)
            d1 = (np.log(s0 / strike) + (r - q + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
            d2 = d1 - vol * np.sqrt(tau)
            return s0 * np.exp(-q * tau) * stats.norm.cdf(d1) \
                - strike * np.exp(-r * tau) * stats.norm.cdf(d2)

    @staticmethod
    def european_option_theta(tau, s0, r, q, vol, strike):
        d1 = EuropeanOption.d1(tau, s0, r, q, vol, strike)
        d2 = EuropeanOption.d2(tau, s0, r, q, vol, strike)
        return strike * np.exp(-r * tau) * stats.norm.cdf(-d2) - s0 * np.exp(-q*tau)*stats.norm.cdf(-d1)

    @staticmethod
    def d1(tau, s0, r, q, vol, strike):
        return np.log(s0 * np.exp((r-q)*tau)/strike)/(vol * np.sqrt(tau)) + 0.5*vol * np.sqrt(tau)

    @staticmethod
    def d2(tau, s0, r, q, vol, strike):
        return EuropeanOption.d1(tau, s0, r, q, vol, strike) - vol * np.sqrt(tau)