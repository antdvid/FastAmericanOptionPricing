import numpy as np
import FastAmericanOptionSolverA as FAOSA
from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import QDplusAmericanOptionSolver as qd


def surface_plot(X,Y,Z,**kwargs):
    """ WRITE DOCUMENTATION
    """
    xlabel, ylabel, zlabel, title = kwargs.get('xlabel',""), kwargs.get('ylabel',""), kwargs.get('zlabel',""), kwargs.get('title',"")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(X,Y,Z,cmap=cm.hot, linewidth=0)
    ax.set_xlabel("r")
    ax.set_ylabel("q")
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.colorbar(surf)
    plt.show()


def random_between(r_min, r_max):
    return r_min + (r_max - r_min) * np.random.rand()

def test():
    # bulk test, randomly set r and q to see the result
    sigma = 0.2  # volatility
    T = 3.0         # maturity

    r_max = 0.1
    r_min = 0.001
    q_max = 0.1
    q_min = 0.0
    S_min = 0.1
    S_max = 200
    K_min = 50
    K_max = 150
    Niters = 500
    option_type = qd.OptionType.Call
    res = []
    count = 0
    total_start_time = time.clock()
    for i in range(Niters):
        ri = random_between(r_min, r_max)
        qi = random_between(q_min, q_max)
        Si = random_between(S_min, S_max)
        Ki = random_between(K_min, K_max)
        solver = FAOSA.FastAmericanOptionSolverA(ri, qi, sigma, Ki, T, option_type)
        solver.max_iters = 15
        solver.iter_tol = 1e-4
        solver.DEBUG = False
        start = time.clock()
        price = solver.solve(0.0, Si)  # t and S
        count += 1
        print("#", count, "r = ", ri, "q = ", qi, "S = ", Si, "K = ", Ki, "price = ", price, "error = ", solver.error, "iters = ", solver.num_iters)
        if math.isnan(price):
            solver.error = 1

        res.append((ri, qi, solver.error, time.clock() - start, solver.num_iters))

    print("total time cost = ", time.clock() - total_start_time, "[s]")
    return res


if __name__=="__main__":
    #with PyCallGraph(output=GraphvizOutput()):
    #    test()
    np.seterr(all="warn", under="ignore")
    res = test()
    r = [t[0] for t in res]
    q = [t[1] for t in res]
    error = [t[2] for t in res]
    time = [t[3] for t in res]
    num_bins = 20
    plt.subplot(1,2,1)
    plt.hist(error, num_bins)
    plt.xlabel("error")
    plt.ylabel("frequency")

    plt.subplot(1,2,2)
    plt.hist(time, num_bins)
    plt.xlabel("time[s]")
    plt.ylabel("frequency")
    plt.show()

    surface_plot(r, q, error)




