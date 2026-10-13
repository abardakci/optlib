from optlib.base import Problem, Optimizer, Result, _default_history
import numpy as np

class GD(Optimizer):
    def __init__(self, step):
        self.step = step
        
    def minimize(self, f: Problem, x0, max_iter=1000, tol=1e-8, **kw) -> Result:
        history = _default_history()
        
        x = np.asarray(x0, dtype=float).ravel().copy()
        fx = None
        converged = False
        n_iter = 0
        while True:
            fx = f(x)
            grad = f.grad(x)
            gradnorm = float(np.linalg.norm(grad))
            
            # Save history
            history["f"].append(fx)
            history["x"].append(x)
            history["gradnorm"].append(gradnorm)
            
            if gradnorm < tol:
                converged = True
                break
            
            if n_iter >= max_iter:
                break
            
            # Descent & Convergence check
            pk = -grad
            x = x + self.step * pk
            history["step"].append(self.step)
            n_iter += 1
            
        return Result(x, fx, n_iter, converged, history)
    
    
class LinearCG(Optimizer):
    def __init__(self):
        pass

    def minimize(self, f, x0, max_iter=1000, tol=1e-8, **kw):
        history=_default_history() # f, gradnorm, x, step
        converged=False
        
        # initial variables for cg
        A = f.A
        b = f.b
        r0 = A @ x0 - b
        rk = r0
        pk = -rk
        xk = x0
        k = 0
        
        while not np.linalg.norm(rk, 2) < tol * np.linalg.norm(r0, 2):
            # update history
            history["x"].append(xk)
            history["f"].append(f(xk))
            history["gradnorm"].append(np.linalg.norm(rk, 2))

            # cg algorithm
            A_pk = A @ pk
            alpha_k = (rk.T @ rk) / (pk.T @ A_pk)
            x_kp1 = xk + alpha_k * pk
            r_kp1 = rk + alpha_k * A_pk
            beta_kp1 = (r_kp1.T @ r_kp1) / (rk.T @ rk)
            p_kp1 = - r_kp1 + beta_kp1 * pk
            
            # update variables
            k += 1
            pk=p_kp1; rk=r_kp1; xk=x_kp1

            # iter limit check
            if k > max_iter:
                break
            
        else:
            converged=True

        history["x"].append(xk)
        history["f"].append(f(xk))
        history["gradnorm"].append(np.linalg.norm(rk, 2))
        return Result(xk, f(xk), k, converged, history)
        