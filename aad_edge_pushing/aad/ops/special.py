# aad/ops/special.py
import numpy as np
from ..core.var import ADVar
from ..core import tape as tape_mod  # Use module access for use_tape() compatibility
from .arithmetic import _as_ad

SQRT_TWO_PI = np.sqrt(2.0 * np.pi)

def norm_pdf(x):
    return np.exp(-0.5 * x * x) / SQRT_TWO_PI

def norm_cdf(x):
    """
    Primitive: returns N(x) and records local partial dN/dx = phi(x).
    We avoid relying on erf being available with AD rules.
    """
    x = _as_ad(x, requires_grad=False)
    # Value: use numpy's erf-based implementation via approximation or NumPy if available
    # Here we use a stable approximation via 0.5*(1+erf(x/sqrt(2))) with np.erf if present.
    try:
        from math import erf, sqrt
        val = 0.5 * (1.0 + erf(x.val / sqrt(2.0)))
    except Exception:
        # fallback: logistic-like smooth approximation (ok for demo)
        val = 1.0 / (1.0 + np.exp(-1.702 * x.val))  # crude, only for demo
    out = ADVar(val)
    pdf = np.exp(-0.5 * x.val * x.val) / SQRT_TWO_PI
    out.dot = pdf * x.dot
    tape_mod.global_tape.push_node(op_tag="norm_cdf", out=out, parents=[(x, norm_pdf(x.val))])
    return out