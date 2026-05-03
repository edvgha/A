from typing import Callable
import numpy as np

def numerical_gradient(f: Callable[[np.ndarray], float], theta: np.ndarray, h=1e-6) -> np.ndarray:
    grad = np.zeros_like(theta)
    it = np.nditer(theta, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original_val = theta[idx]
        
        theta[idx] = original_val + h
        f_plus = f(theta)
        
        theta[idx] = original_val - h
        f_minus = f(theta)
        
        grad[idx] = (f_plus - f_minus) / (2 * h)
        
        theta[idx] = original_val
        it.iternext()
        
    return grad