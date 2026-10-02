from optlib.line_search import BacktrackingLineSearch
from optlib.base import Optimizer, _default_history, Result
from problems.testfuncs import Problem
import numpy as np

def newton(f: Problem, xk: np.ndarray, damping=False, H=None, g=None):
    H = f.hess(xk) if H is None else H
    eig = np.linalg.eigvalsh(H)
    if eig.min() <= 0:
        return None
        
    g = f.grad(xk) if g is None else g
    pk_N = np.linalg.solve(H, -g)
    return pk_N

def bfgs(f: Problem, xk: np.ndarray, H=None, g=None):
    pass

class NewtonOptimizer(BacktrackingLineSearch):
    def __init__(self, rho=0.5, c=0.0001, alpha0=1):        
        def newton_with_damping(f: Problem, xk: np.ndarray):
            H = f.hess(xk)
            L: np.ndarray
            tav = 1.0
            H_ = H
            while True:
                try: 
                    _ = np.linalg.cholesky(H_)
                    break
                except np.linalg.LinAlgError:
                    tav *= 2
                    H_ = H + tav * np.identity(H.shape[0])
                    
            return newton(f, xk, H_)
                
        super().__init__(rho, c, alpha0, direction_method=newton_with_damping)


class BFGS(BacktrackingLineSearch):
    def __init__(self, rho=0.5, c=0.0001, alpha0=1):
        self.Hk: np.ndarray = None
        def bfgs(f: Problem, xk: np.ndarray):
            f_grad_xk = f.grad(xk) 
            n = xk.shape[0]
            self.Hk = np.identity(n, float) if self.Hk is None else self.Hk
            pk = -self.Hk @ f_grad_xk
            return pk
        
        super().__init__(rho, c, alpha0, direction_method=bfgs)

    def minimize(self, f: Problem, x0, max_iter=1000, tol=1e-8, warm_start=False, **kw) -> Result:
        history = _default_history()
        self.Hk=None
        x = np.asarray(x0, dtype=float).ravel().copy()
        fx = None
        converged = False
        n_iter = 0
        alpha = self.alpha0 * self.rho if warm_start else self.alpha0
        
        while True:
            fx = f(x)
            grad = f.grad(x)
            gradnorm = float(np.linalg.norm(grad, 2))

            history["f"].append(fx)
            history["x"].append(x)
            history["gradnorm"].append(gradnorm)

            if gradnorm < tol:
                converged = True
                break

            if n_iter >= max_iter:
                break
                                                
            # compute direction & backtracking
            pk = self.direction_method(f, x)
            
            # warm start or not
            alpha = alpha / self.rho if warm_start else self.alpha0     
            alpha = self._backtracking(f, x, self.c, alpha, pk, self.rho, f_xk=fx, f_grad_xk=grad)
                    
            history["step"].append(alpha)       

            # update
            xk = x
            x = x + alpha * pk
            n_iter += 1
            
            # bfgs condition: Bk must be positive definite
            sk = x - xk
            yk = f.grad(x) - f.grad(xk)
            if not sk @ yk > 0.0:
                raise Exception("[Error] Bk must be positive definite")
            
            sk = sk[:, None]
            yk = yk[:, None]
            rho_k = 1 / (yk.T @ sk)
            n = self.Hk.shape[0]
            I = np.identity(n)
            self.Hk = (I - rho_k * sk @ yk.T) @ self.Hk @ (I - rho_k * yk @ sk.T) + rho_k * sk @ sk.T  
            
        return Result(x, fx, n_iter, converged, history)
