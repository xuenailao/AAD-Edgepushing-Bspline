# aad/ops/transcendental.py
import numpy as np
from ..core.var import ADVar
from ..core import tape as tape_mod  # Use module access for use_tape() compatibility
from .arithmetic import _as_ad

def exp(x):
    x = _as_ad(x, requires_grad=False)
    ex = np.exp(x.val)
    out = ADVar(ex)
    out.dot = ex * x.dot
    tape_mod.global_tape.push_node(op_tag="exp", out=out, parents=[(x, ex)])
    return out

def log(x):
    x = _as_ad(x, requires_grad=False)
    out = ADVar(np.log(x.val))
    out.dot = (1.0 / x.val) * x.dot
    tape_mod.global_tape.push_node(op_tag="log", out=out, parents=[(x, 1.0/x.val)])
    return out

def sqrt(x):
    x = _as_ad(x, requires_grad=False)
    s = np.sqrt(x.val)
    out = ADVar(s)
    out.dot = (0.5 / s) * x.dot
    tape_mod.global_tape.push_node(op_tag="sqrt", out=out, parents=[(x, 0.5/s)])
    return out

def erf(x):
    """
    Error function: erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt

    Derivative: d/dx erf(x) = (2/√π) * e^(-x²)
    """
    x = _as_ad(x, requires_grad=False)
    from scipy.special import erf as scipy_erf

    erf_val = scipy_erf(x.val)
    out = ADVar(erf_val)

    # Derivative: (2/√π) * e^(-x²)
    deriv = (2.0 / np.sqrt(np.pi)) * np.exp(-x.val ** 2)
    out.dot = deriv * x.dot

    tape_mod.global_tape.push_node(op_tag="erf", out=out, parents=[(x, deriv)])
    return out