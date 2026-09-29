import numpy as np
from random import randrange as rng

def grad_check(f: callable, x: np.ndarray, grad: np.ndarray, h=1e-6, n=30, rng=np.random.default_rng(0)):
    assert(x.size == grad.size), "Size mismatch error!"
    
    errors = []
    idx = rng.choice(x.size, n, replace=False)
    for i in idx:
        old = x.flat[i]
        x.flat[i] = old-h; xminus = x.flat[i]; fminus = f(x)
        x.flat[i] = old+h; xplus = x.flat[i]; fplus = f(x)
        x.flat[i] = old
        
        diff = (fplus - fminus) / (xplus - xminus)
        relative_err = abs(grad.flat[i] - diff) / max(abs(grad.flat[i]), abs(diff), 1e-12)
        errors.append((i, relative_err, diff))
    
    return errors
