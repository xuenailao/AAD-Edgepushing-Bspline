# aad/ops/arithmetic.py
import numpy as np
from ..core.var import ADVar
from ..core import tape as tape_mod  # Use module access for use_tape() compatibility

def _as_ad(x, requires_grad=False):
    """Ensure x is an ADVar; otherwise wrap it as a constant ADVar."""
    return x if isinstance(x, ADVar) else ADVar(x, requires_grad=requires_grad)

def _binary(x, y, f, dfdx, dfdy, tag):
    """
    Generic binary primitive:
      - computes out.val = f(x.val, y.val)
      - computes out.dot for FoR JVP when supported by `tag`
      - pushes a Node with local partials (∂out/∂x, ∂out/∂y)
    """
    x = _as_ad(x, requires_grad=False)
    y = _as_ad(y, requires_grad=False)
    out = ADVar(f(x.val, y.val))

    # --------- FoR: JVP (directional derivative) ---------
    if tag == "add":
        out.dot = x.dot + y.dot
    elif tag == "sub":
        out.dot = x.dot - y.dot
    elif tag == "mul":
        # d(x*y) = x.dot * y.val + x.val * y.dot
        out.dot = x.dot * y.val + x.val * y.dot
    elif tag == "div":
        # d(x/y) = (x.dot*y.val - x.val*y.dot) / y.val^2
        out.dot = (x.dot * y.val - x.val * y.dot) / (y.val * y.val)
    else:
        # Default: no explicit JVP rule here (safe fallback for first-order AD)
        try:
            out.dot = np.zeros_like(out.val, dtype=float)
        except Exception:
            out.dot = 0.0
    # ------------------------------------------------------

    tape_mod.global_tape.push_node(
        op_tag=tag, out=out,
        parents=[(x, dfdx(x.val, y.val)), (y, dfdy(x.val, y.val))]
    )
    return out
    # ------------------------------------------------------

def add(x, y): return _binary(x, y, lambda a,b:a+b, lambda a,b:1.0,        lambda a,b:1.0,        "add")
def sub(x, y): return _binary(x, y, lambda a,b:a-b, lambda a,b:1.0,        lambda a,b:-1.0,       "sub")
def mul(x, y): return _binary(x, y, lambda a,b:a*b, lambda a,b:b,          lambda a,b:a,          "mul")
def div(x, y): return _binary(x, y, lambda a,b:a/b, lambda a,b:1.0/b,      lambda a,b:-a/np.square(b),   "div")

def neg(x):
    """
    Unary negation:
      out.val = -x.val
      JVP     : out.dot = -x.dot
    """
    x = _as_ad(x, requires_grad=False)
    out = ADVar(-x.val)
    out.dot = -x.dot
    tape_mod.global_tape.push_node(op_tag="neg", out=out, parents=[(x, -1.0)])
    return out

def pow(x, y):
    """
    Power (demo-level domain handling):
      out.val = x.val ** y.val

    Local partials:
      ∂out/∂x = y * x^(y-1)
      ∂out/∂y = x^y * log(x)        (requires x>0 for non-integer y)

    JVP (FoR):
      If x>0:
        y = x^p
        out.dot = y * ( p' * log(x) + p * x'/x )
      Else (fallback):
        - If p is (near) integer scalar: out.dot ≈ p * x^(p-1) * x'
        - Otherwise: 0 (undefined region for non-integer powers)
    """
    x = _as_ad(x, requires_grad=False)
    y = _as_ad(y, requires_grad=False)

    # primal value
    out = ADVar(x.val ** y.val)

    # JVP (directional derivative)
    xv, pv = x.val, y.val
    yv = out.val
    if np.all(xv > 0):
        logx = np.log(xv)
        out.dot = yv * (y.dot * logx + pv * (x.dot / xv))
    else:
        # fallback handling when x<=0
        try:
            is_int = np.allclose(pv, np.round(pv))
        except Exception:
            is_int = False
        if is_int:
            out.dot = pv * (xv ** (pv - 1.0)) * x.dot
        else:
            # undefined for non-integer exponents; use safe zero
            out.dot = np.zeros_like(yv, dtype=float)

    # local partials for reverse
    dfdx = pv * (xv ** (pv - 1.0))
    dfdy = (xv ** pv) * (np.log(xv) if np.all(xv > 0) else 0.0)

    tape_mod.global_tape.push_node(op_tag="pow", out=out, parents=[(x, dfdx), (y, dfdy)])
    return out

# Bind Python operators to ADVar
ADVar.__add__      = lambda self, other: add(self, other)
ADVar.__radd__     = lambda self, other: add(other, self)
ADVar.__sub__      = lambda self, other: sub(self, other)
ADVar.__rsub__     = lambda self, other: sub(other, self)
ADVar.__mul__      = lambda self, other: mul(self, other)
ADVar.__rmul__     = lambda self, other: mul(other, self)
ADVar.__truediv__  = lambda self, other: div(self, other)
ADVar.__rtruediv__ = lambda self, other: div(other, self)
ADVar.__neg__      = lambda self: neg(self)
ADVar.__pow__      = lambda self, other: pow(self, other)
