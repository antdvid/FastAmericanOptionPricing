from FastAmericanOptionSolverA import *
from multiprocessing import Pool


if __name__ == '__main__':
    # unit test one for valuing American option
    r = 0.04     # risk free
    q = 0.0      # dividend yield
    K = 100       # strike
    S = 80        # underlying spot
    sigma = 0.2  # volatility
    T = 3.0         # maturity
    option_type = qd.OptionType.Put

    solver = FastAmericanOptionSolverA(r, q, sigma, K, T, option_type)
    solver.use_derivative = False
    solver.iter_tol = 1e-5
    solver.max_iters = 20
    price = solver.solve(0.0, S)   # t and S
    print("european put price = ", solver.european_price, "true price = ", 22.0142)
    print("american put price = ", price, "true price = ", 23.22834)

    print("european call price = ", solver.european_price, "true price = ", 4.2758)
    print("american call price = ", price, "true price = ", 4.3948)

    #make a plot for exercise boundary
    plt.plot(solver.shared_tau, solver.shared_B, 'o-')
    plt.plot(solver.shared_tau, solver.shared_B0, 'r--')
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

    plt.figure(3)
    solver.check_f_with_B(np.linspace(100, 500, 50))