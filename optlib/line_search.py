from optlib.base import Problem, Optimizer, Result, _default_history
import numpy as np

def steepest(f, x):
    return -f.grad(x)

class BacktrackingLineSearch(Optimizer):
    def __init__(self, rho=0.5, c=1e-4, alpha0=1.0, direction_method=None):
        self.rho = rho
        self.c = c
        self.alpha0 = alpha0
        self.direction_method = steepest if direction_method is None else direction_method
        
    def _backtracking(self, f, xk, c, alpha, pk, rho, max_tries=32, f_xk=None, f_grad_xk=None):
        """_summary_

        Args:
            f (_type_): function (have the attribute f.grad())
            xk (_type_): initial point (n,)
            c (_type_): armijo constant (0,1)
            alpha (_type_): step length (>0)
            pk (_type_): direction (n,)
            rho (_type_): alpha update param (0,1)
        """
        f_xk = f(xk) if f_xk is None else f_xk
        f_grad_xk = f.grad(xk) if f_grad_xk is None else f_grad_xk
        c_fgrad_pk = c * f_grad_xk @ pk.T
        
        if f_grad_xk.T @ pk >= 0:
            raise Exception("[Error] input pk is not a descending direction")
        
        n_tries = 0
        while not f(xk + alpha * pk) < f_xk + alpha * c_fgrad_pk:
            alpha *= rho
            n_tries += 1
            if n_tries >= max_tries:
                raise Exception(f"[Error] armijo condition could not met in max tries: {max_tries}.")
            
        return alpha
    
    def minimize(self, f: Problem, x0, max_iter=1000, tol=1e-8, warm_start=False, **kw) -> Result:
        history = _default_history()
        
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
            x = x + alpha * pk
            n_iter += 1
            
        return Result(x, fx, n_iter, converged, history)
    