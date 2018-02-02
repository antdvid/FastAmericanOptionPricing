import numpy as np
import FastAmericanOptionSolverA as FAOSA
import cProfile
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

def test():
    # bulk test, randomly set r and q to see the result
    K = 100       # strike
    S = 80        # underlying spot
    sigma = 0.2  # volatility
    T = 3.0         # maturity

    r_max = 0.1
    r_min = 0.01
    q_max = 0.1
    q_min = 0.01
    Niters = 2

    for i in range(Niters):
        ri = r_min + r_max * np.random.rand()
        qi = q_min + q_max * np.random.rand()
        solver = FAOSA.FastAmericanOptionSolverA(ri, qi, sigma, K, T)
        solver.DEBUG = False
        price = solver.solve(0.0, S)  # t and S
        print("r = ", ri, "q = ", qi, "price = ", price, "error = ", solver.error, "iters = ", solver.num_iters)

with PyCallGraph(output=GraphvizOutput()):
    test()

