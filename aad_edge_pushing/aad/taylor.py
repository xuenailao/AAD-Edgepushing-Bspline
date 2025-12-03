# aad/taylor.py
# 2nd-order forward Taylor engine (independent from AAD/tape/edge-pushing)

import math
import numpy as np
from typing import Callable, Dict, Tuple, List


class TVar:
    """
    2nd-order Taylor variable:
    v = v0 + v1 * t + 0.5 * v2 * t^2
    v1 = first directional derivative  (g · d)
    v2 = second directional derivative (d^T H d)
    """
    __slots__ = ("v0", "v1", "v2")

    def __init__(self, v0, v1=0.0, v2=0.0):
        self.v0 = float(v0)
        self.v1 = float(v1)
        self.v2 = float(v2)

    def __add__(a, b):
        if not isinstance(b, TVar): b = TVar(b)
        return TVar(a.v0 + b.v0, a.v1 + b.v1, a.v2 + b.v2)
    __radd__ = __add__

    def __sub__(a, b):
        if not isinstance(b, TVar): b = TVar(b)
        return TVar(a.v0 - b.v0, a.v1 - b.v1, a.v2 - b.v2)

    def __rsub__(b, a):
        if not isinstance(a, TVar): a = TVar(a)
        return TVar(a.v0 - b.v0, a.v1 - b.v1, a.v2 - b.v2)

    def __mul__(a, b):
        if not isinstance(b, TVar): b = TVar(b)
        v0 = a.v0 * b.v0
        v1 = a.v1 * b.v0 + a.v0 * b.v1
        v2 = a.v2 * b.v0 + 2.0 * a.v1 * b.v1 + a.v0 * b.v2
        return TVar(v0, v1, v2)
    __rmul__ = __mul__

    def recip(self):
        x0, x1, x2 = self.v0, self.v1, self.v2
        y0 = 1.0 / x0
        y1 = -x1 / (x0 * x0)
        y2 = 2.0 * (x1 * x1) / (x0 ** 3) - x2 / (x0 * x0)
        return TVar(y0, y1, y2)

    def __truediv__(a, b):
        if not isinstance(b, TVar): b = TVar(b)
        return a * b.recip()

    def __pow__(a, b):
        if not isinstance(b, TVar): b = TVar(b)
        return texp(b * tlog(a))

    def __neg__(a):
        return TVar(-a.v0, -a.v1, -a.v2)


# ----- Elementary functions (2nd order) -----
def texp(x: TVar):
    """Exponential function with 2nd-order Taylor propagation."""
    e = math.exp(x.v0)
    return TVar(e, e * x.v1, e * (x.v2 + x.v1 * x.v1))


def tlog(x: TVar):
    """Natural logarithm with 2nd-order Taylor propagation."""
    x0, x1, x2 = x.v0, x.v1, x.v2
    return TVar(math.log(x0), x1 / x0, x2 / x0 - (x1 * x1) / (x0 * x0))


def tsqrt(x: TVar):
    """Square root with 2nd-order Taylor propagation."""
    r = math.sqrt(x.v0)
    v1 = 0.5 * x.v1 / r
    v2 = 0.5 * (x.v2 / r - 0.5 * x.v1 * x.v1 / (r * x.v0))
    return TVar(r, v1, v2)


def tcos(x: TVar):
    """Cosine with 2nd-order Taylor propagation."""
    s, c = math.sin(x.v0), math.cos(x.v0)
    return TVar(c, -s * x.v1, -c * (x.v1 * x.v1) - s * x.v2)


def tsin(x: TVar):
    """Sine with 2nd-order Taylor propagation."""
    s, c = math.sin(x.v0), math.cos(x.v0)
    return TVar(s, c * x.v1, -s * (x.v1 * x.v1) + c * x.v2)


def norm_pdf0(x0: float):
    """Standard normal PDF at x0."""
    return math.exp(-0.5 * x0 * x0) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: TVar):
    """Standard normal CDF with 2nd-order Taylor propagation."""
    x0, x1, x2 = x.v0, x.v1, x.v2
    phi0 = norm_pdf0(x0)
    Phi0 = 0.5 * (1.0 + math.erf(x0 / math.sqrt(2.0)))
    v0 = Phi0
    v1 = phi0 * x1
    v2 = -x0 * phi0 * (x1 * x1) + phi0 * x2
    return TVar(v0, v1, v2)


def terf(x: TVar):
    """Error function with 2nd-order Taylor propagation."""
    x0, x1, x2 = x.v0, x.v1, x.v2
    erf0 = math.erf(x0)
    # d/dx erf(x) = (2/sqrt(pi)) * exp(-x^2)
    derf = (2.0 / math.sqrt(math.pi)) * math.exp(-x0 * x0)
    # d^2/dx^2 erf(x) = -2x * (2/sqrt(pi)) * exp(-x^2)
    d2erf = -2.0 * x0 * derf
    v0 = erf0
    v1 = derf * x1
    v2 = d2erf * (x1 * x1) + derf * x2
    return TVar(v0, v1, v2)


# Alias names for convenience
exp = texp
log = tlog
sqrt = tsqrt
cos = tcos
sin = tsin
erf = terf


# ----- Main Hessian routine -----
def _taylor_directional(f: Callable[[Dict[str, TVar]], TVar],
                        inputs: Dict[str, float],
                        keys: List[str],
                        d: np.ndarray):
    """
    Evaluate f with directional derivative seed d.
    Returns (f0, f1, f2) = (value, 1st-dir-deriv, 2nd-dir-deriv).
    """
    vars_dict = {k: TVar(inputs[k], d[i], 0.0) for i, k in enumerate(keys)}
    y: TVar = f(vars_dict)
    return y.v0, y.v1, y.v2


def taylor_grad_hessian(f: Callable[[Dict[str, TVar]], TVar],
                        inputs: Dict[str, float]) -> Tuple[Dict[str, float], np.ndarray, float]:
    """
    Compute gradient and Hessian via 2nd-order Taylor propagation.

    Uses n + n(n-1)/2 directional evaluations:
    - n passes for diagonal Hessian entries and gradient
    - n(n-1)/2 passes for off-diagonal entries

    Args:
        f: Function that takes Dict[str, TVar] and returns TVar
        inputs: Input values as Dict[str, float]

    Returns:
        (grad_dict, H, f0):
        - grad_dict: Gradient as Dict[str, float]
        - H: Hessian as np.ndarray of shape (n, n)
        - f0: Function value
    """
    keys = list(inputs.keys())
    n = len(keys)
    grad = np.zeros(n)
    H = np.zeros((n, n))

    # Diagonal passes: e_i direction gives gradient[i] and H[i,i]
    for i in range(n):
        d = np.zeros(n); d[i] = 1.0
        f0, gdir, sdir = _taylor_directional(f, inputs, keys, d)
        grad[i] = gdir
        H[i, i] = sdir

    # Off-diagonal passes: e_i + e_j direction
    # (e_i + e_j)^T H (e_i + e_j) = H[i,i] + 2*H[i,j] + H[j,j]
    # => H[i,j] = 0.5 * (sdir - H[i,i] - H[j,j])
    for i in range(n):
        for j in range(i + 1, n):
            d = np.zeros(n); d[i] = 1.0; d[j] = 1.0
            _, _, sdir = _taylor_directional(f, inputs, keys, d)
            Hij = 0.5 * (sdir - H[i, i] - H[j, j])
            H[i, j] = H[j, i] = Hij

    grad_dict = {k: grad[i] for i, k in enumerate(keys)}
    return grad_dict, H, f0


def taylor_hessian(f: Callable[[Dict[str, TVar]], TVar],
                   inputs: Dict[str, float]) -> np.ndarray:
    """
    Compute only the Hessian via 2nd-order Taylor propagation.

    Convenience wrapper around taylor_grad_hessian.

    Args:
        f: Function that takes Dict[str, TVar] and returns TVar
        inputs: Input values as Dict[str, float]

    Returns:
        H: Hessian as np.ndarray of shape (n, n)
    """
    _, H, _ = taylor_grad_hessian(f, inputs)
    return H
