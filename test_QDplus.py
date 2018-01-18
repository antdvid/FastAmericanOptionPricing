import matplotlib.pyplot as plt
import numpy as np
from QDplusAmericanOptionSolver import *
from EuropeanOptionSolver import *

# unit test one for valuing American option
r = 0.08      # risk free
q = 0.12     # dividend yield
K = 100.0       # strike
S0 = 80.0      # underlying spot
sigma = 0.2  # volatility
tau = 3.0       # maturity

solver = QDplus(r, q, sigma, K)
print("Ep price = ", EuropeanOption.european_option_value(tau, S0, r, q, sigma, K), "Am price =",
      solver.price(tau, S0), "exercise boundary = ", solver.exercise_boundary)
S = np.linspace(1, 2*S0, 100)
plt.plot(S, solver.exercise_boundary_func(S, tau))
plt.plot([0, 2*S0], [0, 0], 'r--')
plt.ylabel("target function")
plt.xlabel("S*")
plt.show()
