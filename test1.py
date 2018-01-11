from FastAmericanOptionSolver import *

# unit test one for valuing American option
r = 0.05      # risk free
q = 0.05      # dividend yield
K = 100       # strike
S = 100       # underlying spot
sigma = 0.25  # volatility
T = 1         # maturity

solver = FastAmericanOptionSolver(r, q, sigma, K, T)
price = solver.solve(0, S)
print("price = ", price)