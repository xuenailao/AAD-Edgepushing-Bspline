# aad/core/__init__.py

"""
Core public API for the AAD package.

This module exposes the minimal set of symbols that users of the AAD framework
should import from `aad.core`. Keeping this surface small makes it easier to
swap/extend internals (e.g., adding FoR/HVP or Edge-Pushing later) without
breaking user code.

Exports:
    ADVar         : The differentiable scalar/tensor wrapper used by the AAD system.
    global_tape   : The default computation graph (tape) used to record ops.
    use_tape      : Context manager to temporarily switch the active tape.
    reverse       : Run a single reverse pass to accumulate first-order adjoints.
    zero_adjoints : Reset all adjoints on the active tape to zero.
    grad          : Convenience: extract the gradient for leaf variables.
    value         : Convenience: extract the primal value(s) from ADVar.
"""

from .var import ADVar
from .tape import global_tape, use_tape
from .engine import reverse, zero_adjoints
from .seeds import grad, value

__all__ = [
    "ADVar",
    "global_tape", "use_tape",
    "reverse", "zero_adjoints",
    "grad", "value",
]
