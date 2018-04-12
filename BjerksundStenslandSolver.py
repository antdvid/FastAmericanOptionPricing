import numpy as np
import scipy.stats as stats

class BksdStld:
    def __init__(self, riskfree, dividend, volatility, strike, maturity):
        self.r = riskfree
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.b = self.r - self.q
        sig_sqr = np.power(self.sigma, 2.0)
        self.beta = (0.5 - self.b/sig_sqr) + np.sqrt(np.power(self.b/sig_sqr - 0.5, 2.0) + 2.0 * self.r / sig_sqr)

    def price(self, S):
        X = self.computeExerciseBoundary()
        alpha = self.computeAlpha(X)
        res = alpha * np.power(S, self.beta) - alpha * self.phi(S,self.T,self.beta, X, X) \
            + self.phi(S, self.T, 1, X, X) - self.phi(S, self.T, 1, self.K, X)\
            - self.K * self.phi(S, self.T, 0, X, X) + self.K * self.phi(S,self.T, 0, self.K, X)
        return res

    def computeAlpha(self, X):
        return (X - self.K) * np.power(X, -self.beta)

    def phi(self, S, T, gamma, H, X):
        sig_sqr = np.power(self.sigma, 2.0)
        lamb = -self.r + gamma * self.b + 0.5 * gamma * (gamma - 1) * sig_sqr
        kappa = 2 * self.b / sig_sqr + (2 * gamma - 1.0)
        d1 = - (np.log(S/H) + (self.b + (gamma - 0.5) * sig_sqr) * T)/(self.sigma * np.sqrt(T))
        d2 = - (np.log(X*X/(S*H)) + (self.b + (gamma - 0.5) * sig_sqr) * T)/(self.sigma * np.sqrt(T))
        res = np.exp(lamb * T) * np.power(S, gamma) * (self.cdf(d1) - pow(X/S, kappa) * self.cdf(d2))
        return res

    @staticmethod
    def cdf(x):
        return stats.norm.cdf(x)

    def computeExerciseBoundary(self):
        B0 = max(self.K, (self.r/(self.r-self.b) * K))
        Binf = self.beta/(self.beta - 1) * self.K
        h = -(self.b * T + 2 * self.sigma * np.sqrt(self.T)) * (pow(self.K, 2.0)/((Binf - B0) * B0))
        return B0 + (Binf - B0) * (1 - np.exp(h))


if __name__ == '__main__':
    r = 0.04  # risk free
    q = 0.04  # dividend yield
    K = 100  # strike
    S0 = 80  # underlying spot
    sigma = 0.2  # volatility
    T = 3.0  # maturity
    call = BksdStld(r, q, sigma, K, T)
    put =  BksdStld(q, r, sigma, S0, T)
    c_price = call.price(S0)
    p_price = put.price(K)
    print("call = ", c_price, ", put = ", p_price)