# aad/core/seeds.py

#-----------------------------------------------------------------------------
# We "plant" a seed (dy/dy = 1) at the scalar output and let gradients grow
# backwards through the tape.
#-----------------------------------------------------------------------------
from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, List, Union
import numpy as np

from .var import ADVar
from .tape import use_tape
from .engine import reverse, zero_adjoints
from .engine import hvp_for as _hvp_for_engine


def value(x: Any) -> Any:
    """Return the numeric value of an ADVar; pass through plain numbers unchanged."""
    return x.val if isinstance(x, ADVar) else x


def _ensure_ad(v: Any, *, name: str, requires_grad: bool = True) -> ADVar:
    """Wrap a plain value as ADVar if needed; otherwise return the ADVar itself."""
    return v if isinstance(v, ADVar) else ADVar(v, requires_grad=requires_grad, name=name)


# ----------------------------- single-input grad ----------------------------- #
def grad(f: Callable[[ADVar], ADVar],
         x0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Gradient of a scalar-output function y=f(x) at x0 (single input).
    Runs one reverse pass within a fresh, isolated tape.
    """
    with use_tape():
        x = _ensure_ad(x0, name="x", requires_grad=True)
        y = f(x)
        if not isinstance(y, ADVar):
            y = ADVar(y, requires_grad=False, name="y")
        # Expect scalar output
        if hasattr(y.val, "shape") and getattr(y.val, "shape", ()) != ():
            raise ValueError("grad(f, x0) expects scalar output.")
        zero_adjoints()
        reverse(y, seed=1.0)
        return x.adj


# ----------------------------- multi-input grads ----------------------------- #
def grads(f: Callable[[Dict[str, ADVar]], ADVar],
          inputs: Dict[str, Union[float, np.ndarray]]) -> Dict[str, Union[float, np.ndarray]]:
    """
    Gradient of a scalar-output function y=f(vars) w.r.t. ALL inputs (dict form).
    Performs ONE reverse pass to obtain all ∂y/∂var simultaneously.

    Parameters
    ----------
    f       : function taking a dict {name: ADVar} and returning a scalar ADVar
    inputs  : dict {name: numeric}

    Returns
    -------
    dict {name: numeric}  # gradients in the same key order as `inputs`
    """
    with use_tape():
        vars_ad: Dict[str, ADVar] = {
            k: _ensure_ad(v, name=k, requires_grad=True) for k, v in inputs.items()
        }
        y = f(vars_ad)
        if not isinstance(y, ADVar):
            y = ADVar(y, requires_grad=False, name="y")
        # Expect scalar output
        if hasattr(y.val, "shape") and getattr(y.val, "shape", ()) != ():
            raise ValueError("grads(f, inputs) expects scalar output.")
        zero_adjoints()
        reverse(y, seed=1.0)
        return {k: vars_ad[k].adj for k in inputs.keys()}


def grads_list(f: Callable[[List[ADVar]], ADVar],
               x0_list: Iterable[Union[float, np.ndarray]]) -> List[Union[float, np.ndarray]]:
    """
    Same as grads(), but the inputs are provided as a list and the result is a list
    of partials in the same order.

    Example
    -------
    f = lambda xs: xs[0]*xs[0] + 3*xs[1]
    grads_list(f, [2.0, 4.0]) -> [4.0, 3.0]
    """
    with use_tape():
        xs: List[ADVar] = [
            _ensure_ad(v, name=f"x{i}", requires_grad=True) for i, v in enumerate(x0_list)
        ]
        y = f(xs)
        if not isinstance(y, ADVar):
            y = ADVar(y, requires_grad=False, name="y")
        # Expect scalar output
        if hasattr(y.val, "shape") and getattr(y.val, "shape", ()) != ():
            raise ValueError("grads_list(f, x0_list) expects scalar output.")
        zero_adjoints()
        reverse(y, seed=1.0)
        return [x.adj for x in xs]


def hvp_for(f, inputs: Dict[str, Union[float, np.ndarray]], v: Dict[str, float]):
    """
    Compute the Hessian-vector product (H·v) for a scalar-output function y = f(vars) using FoR.

    Args
    ----
    f       : function taking a dict {name: ADVar} and returning a scalar ADVar
    inputs  : dict {name: numeric} primal inputs
    v       : dict {name: float} direction vector aligned with `inputs` keys

    Returns
    -------
    np.ndarray with components of H·v in the order of `inputs.keys()`.

    Notes
    -----
    This is a thin wrapper around `engine.hvp_for`. We isolate the computation
    under `with use_tape():` to avoid cross-graph interference.
    """
    with use_tape():
        return _hvp_for_engine(f, inputs, v)
