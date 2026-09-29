from optlib.base import Problem, Optimizer, Result, _default_history
import numpy as np


class BacktrackingLineSearch(Optimizer):
    def __init__(self, rho, c, alpha0):
        self.rho = rho
        self.c = c
        self.alpha0 = alpha0
    
    def _backtracking(self, f, xk, c, alpha, pk, rho):
        """_summary_

        Args:
            f (_type_): function (have the attribute f.grad())
            xk (_type_): initial point (n,)
            c (_type_): armijo constant (0,1)
            alpha (_type_): step length (>0)
            pk (_type_): direction (n,)
            rho (_type_): alpha update param (0,1)
        """
        while not f(xk + alpha * pk) < f(xk) + c * alpha * f.grad(xk) @ pk.T:
            alpha *= rho * alpha
        return alpha     
    
    def minimize(self, f: Problem, x0, max_iter=1000, tol=1e-8, **kw) -> Result:
        history = _default_history()
        
        x = np.asarray(x0, dtype=float).ravel().copy()
        fx = f(x)
        converged = False
        n_iter = 0
        alpha = self.alpha0 * self.rho
        for i in range(max_iter):
            alpha = alpha / self.rho # warm start
            n_iter = i
            grad = f.grad(x)
            gradnorm = float(np.linalg.norm(grad))
            
            # Save history
            history["f"].append(fx)
            history["x"].append(x)
            history["gradnorm"].append(gradnorm)
            history["step"].append(alpha)       
            
            if gradnorm < tol:
                converged = True
                break
            
            # Descent & Convergence check
            alpha = self._backtracking(f, x, self.c, alpha, -grad, self.rho)
            x = x - alpha * grad
            fx = f(x)
            
        return Result(x, fx, n_iter, converged, history)
    