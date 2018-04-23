import matplotlib.pyplot as plt
import numpy as np
from QDplusAmericanOptionSolver import *
from EuropeanOptionSolver import *

if __name__ == '__main__':
    # unit test one for valuing American option
    r = 0.04  # risk free
    q = 0.0  # dividend yield
    K = 100  # strike
    S0 = 80  # underlying spot
    sigma = 0.2  # volatility
    T = 3.0  # maturity
    option_type = OptionType.Put
    tau = 0.5

    solver = QDplus(r, q, sigma, K, option_type)
    print("Ep call true price = 4.2758, Am call true price = 4.3948")
    print("Ep put true price = 22.0142, Am put true price = 23.22834")
    print("Ep call price = ", EuropeanOption.european_call_value(tau, S0, r, q, sigma, K))
    print("Ep put price = ", EuropeanOption.european_put_value(tau, S0, r, q, sigma, K))
    print("Am price =", solver.price(tau, S0), "exercise boundary = ", solver.exercise_boundary)

    S = np.linspace(1, 4*S0, 200)
    plt.plot(S, solver.exercise_boundary_func(S, tau), 'o-')
    plt.plot([0, 4*S0], [0, 0], 'r--')
    plt.ylim([-2*K, 2 * K])
    plt.ylabel("target function")
    plt.xlabel("S*")
    plt.show()
