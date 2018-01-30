import numpy as np
import FastAmericanOptionSolverB as solver
import matplotlib.pyplot as plt
from matplotlib import *
import matplotlib


r = 0.05      # risk free
q = 0.05      # dividend yield
K = 100       # strike
S = 100        # underlying spot
sigma = 0.2  # volatility
T = 1.5         # maturity
t = 0.0         # valuation date
Nt = 15
Nb = 50

mysolver = solver.FastAmericanOptionSolver(r, q, sigma, K, T)
mysolver.set_collocation_points()
mysolver.set_initial_guess()
Bspace = np.linspace(np.min(mysolver.shared_B), np.max(mysolver.shared_B), Nb)
tauspace = np.linspace(0.1,1.5,Nt)

plt.plot(mysolver.shared_tau, mysolver.shared_B, "o-")
plt.xlabel("tau [Y]")
plt.ylabel("Exercise boundary [$]")
plt.show()

norm = matplotlib.colors.Normalize(vmin=0,vmax=T)

# choose a colormap
c_m = matplotlib.cm.hot

# create a ScalarMappable and initialize a data structure
s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
s_m.set_array([])

for tau in tauspace:
    print("tau = ", tau)
    B = Bspace
    fprime = []
    f = []
    for Bi in Bspace:
        res = mysolver.compute_f_and_fprime(tau, Bi)
        fprime.append(res[1])
        f.append(res[0])
    plt.subplot(1, 2, 1)
    plt.plot(B, f, '-', color=s_m.to_rgba(tau))
    plt.xlabel("B")
    plt.ylabel("f")
    plt.subplot(1, 2, 2)
    plt.plot(B, fprime, '-', color=s_m.to_rgba(tau))

plt.colorbar(s_m)
plt.xlabel("B")
plt.ylabel("f prime")
plt.show()
