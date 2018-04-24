import matplotlib.pyplot as plt
import numpy as np
from QDplusAmericanOptionSolver import *
from EuropeanOptionSolver import *

if __name__ == '__main__':
    # unit test one for valuing American option
    r = 0.0975729097939295     # risk free
    q = 0.011804520625954162      # dividend yield
    K = 105.61782314803582       # strike
    S0 = 30.543317986992072        # underlying spot
    sigma = 0.2  # volatility
    T = 3  # maturity
    option_type = OptionType.Put

    tau = [3.00000000e+00,   2.89864827e+00,   2.61153811e+00,   2.18566017e+00,
             1.68750000e+00,   1.18846904e+00,   7.50000000e-01,   4.12011906e-01,
             1.87500000e-01,   6.43398282e-02,   1.34618943e-02,   8.70786986e-04,
             0.00000000e+00]

    solver = QDplus(r, q, sigma, K, option_type)

    B_tau = []
    for tau_i in tau:
        exercise_boundary = solver.compute_exercise_boundary(tau_i)
        B_tau.append(exercise_boundary)

    print("tau = ", tau)
    print("Btau = ", B_tau)
    plt.plot(tau, B_tau, 'o-')
    plt.show()
