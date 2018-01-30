import FastAmericanOptionSolverA as MA
import FastAmericanOptionSolverB as MB
import matplotlib.pyplot as plt
import numpy as np

# unit test one for valuing American option
r = 0.045      # risk free
q = 0.05      # dividend yield
K = 130       # strike
S = 100        # underlying spot
sigma = 0.25  # volatility
T = 1         # maturity

solver = MA.FastAmericanOptionSolverA(r, q, sigma, K, T)
price = solver.solve(0.0, S)   # t and S
plt.figure()
iters = np.array([float(x[0]) for x in solver.iter_records])
errors = np.array([x[1] for x in solver.iter_records])
plt.loglog(iters, errors, 'o-')

solver = MB.FastAmericanOptionSolver(r, q, sigma, K, T)
price =solver.solve(0.0, S)
iters = np.array([float(x[0]) for x in solver.iter_records])
errors = np.array([x[1] for x in solver.iter_records])
plt.loglog(iters, errors, 'o-')

solver = MA.FastAmericanOptionSolverA(r, q, sigma, K, T)
solver.use_derivative = True
price = solver.solve(0.0, S)   # t and S
iters = np.array([float(x[0]) for x in solver.iter_records])
errors = np.array([x[1] for x in solver.iter_records])
plt.loglog(iters, errors, 'o-')

solver = MA.FastAmericanOptionSolver(r, q, sigma, K, T)
solver.use_derivative = True
price = solver.solve(0.0, S)   # t and S
iters = np.array([float(x[0]) for x in solver.iter_records])
errors = np.array([x[1] for x in solver.iter_records])
plt.loglog(iters, errors, 'o-')

plt.legend(["Method A", "Method B", "Method A with derivative", "Method B with derivative"])
plt.xlabel("Number of iterations")
plt.ylabel("Match condition error")
plt.show()