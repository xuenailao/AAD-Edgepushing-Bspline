# aad/core/var.py
from __future__ import annotations
import numpy as np
from typing import Any, Optional

class ADVar:
    """
    Active variable for reverse-mode Automatic Differentiation (AD).

    Attributes
    ----------
    val : float | np.ndarray
        Forward (primal) value of this variable.
    adj : float | np.ndarray
        Reverse-mode adjoint (gradient accumulator); same shape as val.
    dot : float | np.ndarray
        Forward tangent (directional derivative for JVP). Used in FoR.
    adj_dot : float | np.ndarray
        Reverse-mode companion for the tangent (directional adjoint). Used in FoR.
    requires_grad : bool
        Whether this variable participates in differentiation. If False,
        the variable is treated as constant (no parents recorded on tape).
    name : Optional[str]
        Optional debug/pretty-print name.
    """

    __array_priority__ = 1000  # ensures NumPy ufuncs prefer ADVar.__array_ufunc__

    def __init__(self, val: Any, *, requires_grad: bool = True, name: Optional[str] = None):
        # Type check: only allow numeric scalars, sequences, or numpy arrays
        if not isinstance(val, (int, float, list, tuple, np.ndarray)):
            raise TypeError(
                f"ADVar only accepts numeric types (int, float, list, tuple, ndarray), "
                f"but got {type(val)}"
            )

        # Convert input to numeric form with explicit float64 for precision:
        # - list/tuple/ndarray → numpy float64 array
        # - int/float → numpy float64 scalar
        if isinstance(val, (list, tuple, np.ndarray)):
            self.val = np.asarray(val, dtype=np.float64)
        else:
            self.val = np.float64(val)

        # Gradient accumulator (adjoint), initialized to zeros with same shape as val
        self.adj = np.zeros_like(self.val, dtype=float)

        # --- Additional FoR fields (for second-order differentiation) ---
        # Forward tangent (JVP seed/propagation); same shape as val
        self.dot = np.zeros_like(self.val, dtype=float)
        # Reverse companion for tangent (accumulates Hessian-vector contributions); same shape
        self.adj_dot = np.zeros_like(self.val, dtype=float)

        # Flag: whether to track this variable for gradients
        self.requires_grad = requires_grad

        # Optional user-provided name (for debugging / display)
        self.name = name

    def __repr__(self):
        # Short label: "req" if requires_grad=True, else "const"
        rg = "req" if self.requires_grad else "const"
        return f"ADVar({self.val!r}, {rg}, name={self.name!r})"

    # Operator overloading for arithmetic operations
    def __add__(self, other):
        from ..ops.arithmetic import add
        return add(self, other)

    def __radd__(self, other):
        from ..ops.arithmetic import add
        return add(other, self)

    def __sub__(self, other):
        from ..ops.arithmetic import sub
        return sub(self, other)

    def __rsub__(self, other):
        from ..ops.arithmetic import sub
        return sub(other, self)

    def __mul__(self, other):
        from ..ops.arithmetic import mul
        return mul(self, other)

    def __rmul__(self, other):
        from ..ops.arithmetic import mul
        return mul(other, self)

    def __truediv__(self, other):
        from ..ops.arithmetic import div
        return div(self, other)

    def __rtruediv__(self, other):
        from ..ops.arithmetic import div
        return div(other, self)

    def __neg__(self):
        from ..ops.arithmetic import neg
        return neg(self)

    def __pow__(self, other):
        from ..ops.arithmetic import pow
        return pow(self, other)

    def __rpow__(self, other):
        from ..ops.arithmetic import pow
        return pow(other, self)
