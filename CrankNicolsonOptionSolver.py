import numpy as np
import matplotlib.pyplot as plt


class CNOptionSolver:
    def __init__(self, riskfree, dividend, volatility, strike, maturity):
        self.r = riskfree
        self.q = dividend
        self.sigma = volatility
        self.K = strike
        self.T = maturity
        self.Smax = 3 * strike
        self.S = []
        self.X = []
        self.A = []
        self.b = []
        self.N = 200
        self.max_dt = 1/12.0
        self.USE_PSOR = False
        self.tol = 1e-5
        self.max_iter = 200
        self.omega = 1.2
        self.cached_dt = 0
        self.err = 0
        self.iter = 0

    def solve(self, S0):
        self.setInitialCondition()

        self.solvePDE()

        x = self.S.flatten()
        y = self.X.flatten()
        return np.interp(S0, x, y)

    def solvePDE(self):
        t = self.T
        while t > 0:
            dt = min(t, self.max_dt)
            self.setCoeff(dt)
            if self.USE_PSOR:
                self.solvePSOR()
            else:
                self.solveLinearSystem()
            t -= dt
            print("t = ", t, " err = ", self.err, "iters = ", self.iter)

    def setInitialCondition(self):
        self.S = np.linspace(0, self.Smax, self.N)
        self.A = np.zeros((self.N, self.N))
        self.b = np.zeros((self.N, 1))
        self.X = np.maximum(self.K - self.S, 0)


    def setCoeff(self, dt):
        N = self.N
        r = self.r
        q = self.q
        S = self.S
        X = self.X
        sigma = self.sigma
        dS = S[1] - S[0]
        for i in range(0, N-1):
            alpha = 0.25 * dt * (np.square(sigma*S[i]/dS) - (r - q) * S[i]/dS)
            beta = 0.5 * dt * (r + np.square(sigma * S[i]/dS))
            gamma = 0.25 * dt * (np.square(sigma*S[i]/dS) + (r - q) * S[i]/dS)
            if i == 0:
                self.b[i] = X[i] * (1 - beta)
                self.A[i][i] = 1 + beta
            else:
                self.b[i] = alpha * X[i-1] + (1 - beta) * X[i] + gamma * X[i+1]
                self.A[i][i-1] = -alpha
                self.A[i][i] = 1 + beta
                self.A[i][i+1] = -gamma
        self.A[-1][N-4] = -1
        self.A[-1][N-3] = 4
        self.A[-1][N-2] = -5
        self.A[-1][N-1] = 2
        self.b[-1] = 0

    def solveLinearSystem(self):
        self.X = np.linalg.solve(self.A, self.b)

    def solvePSOR(self):
        N = self.N
        iter = 0
        omega = self.omega
        self.err = 1000
        while self.err > self.tol and iter < self.max_iter:
            iter += 1
            x_old = self.X.copy()
            for i in range(N-1):
                self.X[i] = (1 - omega) * self.X[i] + omega * self.b[i] / self.A[i][i]
                self.X[i] -= self.A[i][i+1] * self.X[i+1] * omega / self.A[i][i]
                self.X[i] -= self.A[i][i-1] * self.X[i-1] * omega / self.A[i][i]

            #for last row, use boundary condition
            self.X[N-1] = (1 - omega) * self.X[i] + omega * self.b[i] / self.A[i][i]
            for j in range(N-4, N):
                self.X[N-1] -= self.A[N-1][j] * self.X[j] * omega / self.A[N-1][N-1]

            self.applyConstraint()
            self.err = np.linalg.norm(x_old - self.X)
            self.iter = iter

    def applyConstraint(self):
        self.X = np.maximum(self.X, self.K - self.S)

if __name__ == '__main__':
    # unit test one for valuing American option
    r = 0.04  # risk free
    q = 0.04  # dividend yield
    K = 100  # strike
    S = 80  # underlying spot
    sigma = 0.2  # volatility
    T = 3.0  # maturity
    solver = CNOptionSolver(r, q, sigma, K, T)
    solver.N = 800
    solver.max_dt = 0.01
    solver.USE_PSOR = True
    price = solver.solve(S)
    print(price)

    x = solver.S.flatten()
    y = solver.X.flatten()
    plt.plot(x,y)
    plt.show()

