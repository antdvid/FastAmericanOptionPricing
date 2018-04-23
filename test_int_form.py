import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from EuropeanOptionSolver import *
from scipy.integrate import quad
import FastAmericanOptionSolverBase as fao
import FastAmericanOptionSolverB as faoB
import QDplusAmericanOptionSolver as qd

def dminus(tau, z):
    return (np.log(z) + (r - q) * tau - 0.5 * sigma * sigma * tau) / (sigma * np.sqrt(tau))

def dplus(tau, z):
    return (np.log(z) + (r - q) * tau + 0.5 * sigma * sigma * tau) / (sigma * np.sqrt(tau))

def integrand(u):
    return q * S * np.exp(-q * (tau - u)) * stats.norm.cdf(dplus(tau - u, S/B(u))) \
           - r * K * np.exp(-r * (tau - u)) * stats.norm.cdf(dminus(tau - u, S/B(u)))

def B(u):
    Bi = [162.41787321309212, 161.70263557937821, 159.55982941639562, 156.0063773574426, 151.09559908000622,
             144.94755881663042, 137.7786228627495, 129.92122783910395, 121.82751657176807, 114.05701310124692,
             107.26119495030882, 102.20484830383452, 100]

    taui = [3.00000000e+00,   2.89864827e+00,   2.61153811e+00,   2.18566017e+00,
           1.68750000e+00,   1.18846904e+00,   7.50000000e-01,   4.12011906e-01,
           1.87500000e-01,   6.43398282e-02,   1.34618943e-02,   8.70786986e-04,
           0.00000000e+00]

    yinterp = np.interp([u], taui, Bi)
    return yinterp[0]

if __name__ == '__main__':
    # unit test one for valuing American option
    r = 0.04      # risk free
    q = 0.04      # dividend yield
    K = 100       # strike
    S = 80        # underlying spot
    sigma = 0.2  # volatility
    T = 3.0         # maturity
    tau = T

    eur_call = EuropeanOption.european_call_value(tau, S, r, q, sigma, K)
    print("eur call = ", eur_call)

    #prem = quad(integrand, 0, tau)[0]

    #print("prem = ", prem)

    solver = faoB.FastAmericanOptionSolverB(r, q, sigma, K, T, qd.OptionType.Call)
    solver.max_iters = 0
    solver.DEBUG = True
    price = solver.solve(0, S)
    print("am call price = ", price, "eur call price = ", solver.european_price)

    plt.plot(solver.shared_tau, solver.shared_B, 'b-s')

    solver.compute_integration_terms(tau, 10)

    plt.plot(solver.shared_u, solver.shared_Bu, 'g-o')


    plt.show()

    # Bu_test = np.array([162.41787321309212, 161.70263557937821, 159.55982941639562, 156.0063773574426, 151.09559908000622,
    #          144.94755881663042, 137.7786228627495, 129.92122783910395, 121.82751657176807, 114.05701310124692,
    #          107.26119495030882, 102.20484830383452, 100])
    #
    # u_test = np.array([3.00000000e+00,   2.89864827e+00,   2.61153811e+00,   2.18566017e+00,
    #        1.68750000e+00,   1.18846904e+00,   7.50000000e-01,   4.12011906e-01,
    #        1.87500000e-01,   6.43398282e-02,   1.34618943e-02,   8.70786986e-04,
    #        0.00000000e+00])
    # integrand_u = []
    # for ui, Bui in zip(u_test, Bu_test):
    #     integrand_u.append(solver.v_integrand_12(tau, S, ui, Bui))
    #
    # u_test2 = solver.shared_u
    # Bu_test2 = solver.shared_Bu
    # integrand_u = []
    # for ui, Bui in zip(u_test2, Bu_test2):
    #     integrand_u.append(solver.v_integrand_12(tau, S, ui, Bui))
    # print(u_test2)
    # print(Bu_test2)
    #
    # plt.plot(u_test, Bu_test, 'b-o', u_test2, Bu_test2, 'r-*', solver.shared_u, solver.shared_Bu, 'k-s')
    # plt.show()
    #
    # premium = solver.quadrature_sum(solver.v_integrand_12, tau, S, solver.quadrature_num)
    # print("premium = ", premium)
    # print(integrand_u)
    #
    # # u = np.linspace(0, 3.0, 100)
    # # plt.plot(u_test, integrand_u, '*-')
    # # plt.show()

