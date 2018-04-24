from FastAmericanOptionSolverB import *

# unit test one for valuing American option
r = 0.04      # risk free
q = 0.0      # dividend yield
K = 100       # strike
S = 80        # underlying spot
sigma = 0.2  # volatility
T = 3.0         # maturity
option_type = qd.OptionType.Call

solver = FastAmericanOptionSolverB(r, q, sigma, K, T, option_type)
solver.use_derivative = False
solver.DEBUG = True
solver.max_iters = 20
price = solver.solve(0.0, S)   # t and S
print("european price = ", solver.european_price)
print("american price = ", price)

#make a plot for exercise boundary
plt.plot(solver.shared_tau, solver.shared_B, 'o-')
plt.plot(solver.shared_tau, solver.shared_B0, '*-')
plt.legend(["real exercise boundary", "initial guess"])
plt.xlabel("Time to maturity tau")
plt.ylabel("Exercise boundary [$]")
plt.show()

plt.figure(2)
iters = np.array([float(x[0]) for x in solver.iter_records])
errors = np.array([x[1] for x in solver.iter_records])
plt.loglog(iters, errors, 'o-')
plt.xlabel("Number of iterations")
plt.ylabel("Match condition error")
plt.show()