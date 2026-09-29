from optlib.base import Problem, Optimizer, Result, _default_history
import numpy as np

class GD(Optimizer):
    def __init__(self, step):
        self.step = step
        
    def minimize(self, f: Problem, x0, max_iter=1000, tol=1e-8, **kw) -> Result:
        history = _default_history()
        
        x = np.asarray(x0, dtype=float).ravel().copy()
        fx = f(x)
        converged = False
        n_iter = 0
        for i in range(max_iter):
            n_iter = i
            grad = f.grad(x)
            gradnorm = float(np.linalg.norm(grad))
            
            # Save history
            history["f"].append(fx)
            history["x"].append(x)
            history["gradnorm"].append(gradnorm)
            history["step"].append(self.step)       
            
            if gradnorm < tol:
                converged = True
                break
            
            # Descent & Convergence check
            x = x - self.step * grad
            fx = f(x)
            
        return Result(x, fx, n_iter, converged, history)
    