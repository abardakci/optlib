from dataclasses import dataclass, field
from typing import Any

class Problem:    
    def __call__(self, x):
        raise NotImplementedError
    
    def grad(self, x):
        raise NotImplementedError
    
    def hess(self, x): 
        return None


def _default_history():
    return {
        "f": [], "gradnorm": [], "x": [], "step": []
    }

@dataclass
class Result:
    x: Any
    fval: float
    n_iter: int
    converged: bool
    history: dict[str, list] = field(default_factory=_default_history)


class Optimizer:
    def minimize(self, f, x0, max_iter=1000, tol=1e-8, **kw) -> Result:
        raise NotImplementedError


# phi(alpha) = f(xk + alpha * pk)
class Phi:
    """A function wrapper, for single dim outputs
    nfev: number of func evaluation
    ngev: number of grad evaluation
    _fcache, _gcache: cache mechanism to avoid recalculating grad or func value 
    """

    def __init__(self, f, xk, pk, f_xk=None, g_xk=None):
        self.f, self.xk, self.pk = f, xk, pk
        self.nfev = 0
        self.ngev = 0
        self._fcache = {0.0: f(xk) if f_xk is None else f_xk}
        self._gcache = {0.0: f.grad(xk) if g_xk is None else g_xk}
        self.nfev += f_xk is None
        self.ngev += g_xk is None
        self.phi0 = float(self._fcache[0.0])
        self.dphi0 = float(self._gcache[0.0] @ pk)

    def x(self, alpha):
        return self.xk + alpha * self.pk

    def __call__(self, alpha):
        alpha = float(alpha)
        if alpha not in self._fcache:
            self._fcache[alpha] = float(self.f(self.x(alpha)))
            self.nfev += 1
        return self._fcache[alpha]

    def d(self, alpha):
        alpha = float(alpha)
        if alpha not in self._gcache:
            self._gcache[alpha] = self.f.grad(self.x(alpha))
            self.ngev += 1
        return float(self._gcache[alpha] @ self.pk)

    def grad_at(self, alpha):
        return self._gcache.get(float(alpha))
