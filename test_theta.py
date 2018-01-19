import EuropeanOptionSolver as eurp
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1e-10, 5, 2000)
theta = eurp.EuropeanOption.european_option_theta(x, 70, 0.05, 0.05, 0.2, 100)
plt.ylim([-5, 2])
plt.plot(x, theta)
plt.show()
