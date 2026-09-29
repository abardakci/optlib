import numpy as np
from optlib.base import Problem

class Quadratic(Problem):
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b
    
    def __call__(self, x: np.ndarray):
        return 1/2 * (x.T @ self.A @ x) - self.b.T @ x
    
    def grad(self, x: np.ndarray):
        return self.A @ x - self.b
    
    def hess(self):
        return self.A


def make_quadratic(n: int, kappa: int, seed=0):
    """returns n x n SPD A matrix whose cond=kappa"""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n,n), dtype=np.float64)
    Q, _ = np.linalg.qr(M)
    eigs = np.logspace(start=0, stop=np.log10(kappa), num=n)
    return Q.T @ np.diag(eigs) @ Q


class Rosenbrock2d(Problem):
    def __init__(self, a: np.float64, b: np.float64):
        self.a = a
        self.b = b
    
    def __call__(self, x: np.ndarray):
        x, y = x[0], x[1]
        a, b = self.a, self.b
        return (a - x)**2 + b*(y - x**2)**2
    
    def grad(self, x: np.ndarray):
        x, y = x[0], x[1]
        a, b = self.a, self.b
        return np.array([(-2 * (a - x)) + (2 * b*(y - x**2) * (-2*x)), 
                         2 * b*(y - x**2)])
    
    def hess(self, x: np.ndarray):
        x, y = x[0], x[1]
        a, b = self.a, self.b
        df_dx = 2.0 + 4.0 * b*(3.0 * x**2 - y), -4.0*b*x 
        df_dy = 2.0*b, 2.0*b
        return np.array([df_dx, df_dy], dtype=np.float64)


class RosenbrockNd(Problem):
    def __init__(self):
        pass
    
    def __call__(self, x: np.ndarray):
        N = x.size
        accum = 0
        for i in range(0, N-1):
            accum += (100.0*(x[i+1] - x[i]**2)**2 + (1.0-x[i])**2)
        return accum
        
    def grad(self, x: np.ndarray):
        g = np.zeros_like(x, dtype=np.float64)
        g[:-1] = -400.0*x[:-1]*(x[1:] - x[:-1]**2) - 2.0*(1.0 - x[:-1])
        g[1:] += 200.0*(x[1:] - x[:-1]**2)
        return g
    
    def hess(self, x: np.ndarray):
        n = x.size
        H = np.zeros(shape=(n, n), dtype=np.float64)
        
        d = np.zeros(n, dtype=np.float64)
        d[:-1] = 1200.0 * x[:-1]**2 - 400 * x[1:] + 2.0
        d[1:] += 200.0
        np.fill_diagonal(H, d)
        
        off = -400.0 * x[:-1]
        i = np.arange(n-1)
        H[i, i+1] = off
        H[i+1, i] = off
        return H
        

class Himmelblau(Problem):
    def __init__(self):
        pass

    def __call__(self, x: np.ndarray):
        x, y = x[0], x[1]
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    def grad(self, x: np.ndarray):
        x, y = x[0], x[1]
        df_dx = 4*x * (x**2+y-11) + 2*(x + y**2 -7)
        df_dy = 2*(x**2 + y - 11) + 4*y * (x + y**2 - 7) 
        return np.array([df_dx, df_dy], dtype=np.float64)
        
    def hess(self, x: np.ndarray):
        x, y = x[0], x[1]
        df2_dx2 = 4*(x**2+y-11)+8*x**2 + 2
        df2_dy2 = 2 + 4*(x+y**2-7) + 8 * y**2
        df2_dxy = 4*y*(x+1)
        return np.array([[df2_dx2, df2_dxy], [df2_dxy, df2_dy2]])
    