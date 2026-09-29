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
