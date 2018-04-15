import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from FastAmericanOptionSolverA import *
from FastAmericanOptionSolverB import *


if __name__ == '__main__':
    # unit test one for valuing American option
    r = 0.04      # risk free
    q = 0.04      # dividend yield
    K = 100       # strike
    S = 80        # underlying spot
    sigma = 0.2  # volatility
    T = 3.0         # maturity
    option_type = qd.OptionType.Call

    solver = FastAmericanOptionSolverB(r, q, sigma, K, T, option_type)
    solver.iter_tol = 1e-3
    solver.max_iters = 20
    solver.use_derivative = False
    solver.solve(0, S)
    solver.check_f_with_B(np.linspace(95, 200, 200))