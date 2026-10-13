from optlib.line_search import BacktrackingLineSearch, backtracking
from optlib.base import Optimizer, _default_history, Result
from problems.testfuncs import Problem
import numpy as np
import scipy
from scipy.linalg import solve_triangular
from functools import partial

def newton(f: Problem, xk: np.ndarray, damping_on=False, H=None, g=None):
    H = f.hess(xk) if H is None else H
    g = f.grad(xk) if g is None else g

    L: np.ndarray=None
    H_: np.ndarray = H
    tav: float = 0.0
    try:
        L = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        if damping_on:
            H_, L, tav = damping(f, xk, H)
        
    if L is not None:
        z = solve_triangular(L, -g, lower=True)
        pk_N = solve_triangular(L.T, z, lower=False)
    else:
        pk_N = np.linalg.solve(H, -g)
    
    return pk_N, {"tav": tav}


def damping(f: Problem, xk: np.ndarray, H: np.ndarray, tav=1.0):
    L: np.ndarray=None
    tav = tav

    H_ = H + tav * np.identity(H.shape[0])
    for i in range(100):
        try:
            L = np.linalg.cholesky(H_)
            break
        except np.linalg.LinAlgError:                
            tav *= 2.0
            H_ = H + tav * np.identity(H.shape[0])
            
    return H_, L, tav


def bfgs(f: Problem, xk: np.ndarray, Hk: np.ndarray, f_grad_xk=None):
    f_grad_xk = f.grad(xk) if f_grad_xk is None else f_grad_xk
    n = xk.shape[0]
    Hk = np.identity(n, float) if Hk is None else Hk
    pk = -Hk @ f_grad_xk
    return pk, Hk


class NewtonOptimizer(BacktrackingLineSearch):
    def __init__(self, rho=0.5, c=0.0001, alpha0=1, damping=False):
        nwtn_dir = partial(newton, damping_on=damping)
        super().__init__(rho, c, alpha0, direction_method=nwtn_dir)


class BFGS(BacktrackingLineSearch):
    def __init__(self, rho=0.5, c=0.0001, alpha0=1):
        self.Hk: np.ndarray = None
        super().__init__(rho, c, alpha0, direction_method=bfgs)

    def minimize(self, f: Problem, x0, max_iter=1000, tol=1e-8, warm_start=False, **kw) -> Result:
        history = _default_history()
        self.Hk=None
        xk = np.asarray(x0, dtype=float).ravel().copy()
        f_xk = None
        f_xkp1 = None
        converged = False
        n_iter = 0
        alpha = self.alpha0 * self.rho if warm_start else self.alpha0
        
        f_grad_xk: np.ndarray = None
        f_grad_xkp1: np.ndarray = None
        while True:
            f_xk = f(xk) if f_xkp1 is None else f_xkp1 
            f_grad_xk = f.grad(xk) if f_grad_xkp1 is None else f_grad_xkp1
            gradnorm = float(np.linalg.norm(f_grad_xk, 2))

            history["f"].append(f_xk)
            history["x"].append(xk)
            history["gradnorm"].append(gradnorm)

            if gradnorm < tol:
                converged = True
                break

            if n_iter >= max_iter:
                break
                                                
            # compute direction & backtracking
            pk, self.Hk = self.direction_method(f, xk, self.Hk)
            
            # warm start or not
            alpha = alpha / self.rho if warm_start else self.alpha0     
            alpha, xkp1, f_xkp1,  = backtracking(f, xk, self.c, alpha, pk, self.rho, f_xk=f_xk, f_grad_xk=f_grad_xk)
                    
            history["step"].append(alpha)       

            # update
            f_xkp1 = xk + alpha * pk
            n_iter += 1
            
            # bfgs condition: Bk must be positive definite
            sk = f_xkp1 - xk
            yk = f_grad_xkp1 - f_grad_xk # grad(x_(k+1)) - grad(x_k)
            if not sk @ yk > 0.0:
                raise Exception("[Error] Bk must be positive definite")
            
            sk = sk[:, None]
            yk = yk[:, None]
            rho_k = 1 / (yk.T @ sk)
            n = self.Hk.shape[0]
            I = np.identity(n)
            self.Hk = (I - rho_k * sk @ yk.T) @ self.Hk @ (I - rho_k * yk @ sk.T) + rho_k * sk @ sk.T  
            
        return Result(xk, f_xk, n_iter, converged, history)
