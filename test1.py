from FastAmericanOptionSolver import *

# unit test one for valuing American option
r = 0.04      # risk free
q = 0.04      # dividend yield
K = 100       # strike
S = 80       # underlying spot
sigma = 0.2  # volatility
T = 3.0         # maturity

solver = FastAmericanOptionSolver(r, q, sigma, K, T)
price = solver.solve(0.0, S)   # t and S
plt.plot(solver.shared_tau, solver.shared_B, 'o-')
plt.show()
print("european price = ", solver.european_put_price)
print("price = ", price, "true price = ", 23.22834)