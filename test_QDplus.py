import matplotlib.pyplot as plt
import numpy as np
from QDplusAmericanOptionSolver import *

# unit test one for valuing American option
r = 0.05      # risk free
q = 0.02      # dividend yield
K = 100       # strike
S0 = 100       # underlying spot
sigma = 0.25  # volatility
T = 1         # maturity

solver = QDplus(r, q, sigma, K, T)
print("price1 =",  solver.price(T, S0))
# S = np.linspace(1, 2*S0, 100)
# plt.plot(S, solver.exercise_boundary_func(S, T))
# plt.plot([0, 2*S0], [0, 0], 'r--')
# plt.show()
#print("price2 =",  solver.exercise_boundary_func(200, T))
