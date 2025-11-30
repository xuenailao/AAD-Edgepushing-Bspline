# aad/ops/__init__.py

# Ensure operator overloading is registered
from . import arithmetic
from . import transcendental
from . import special

# Convenience re-exports so users can do: from aad.ops import mul, exp, ...
from .arithmetic import add, sub, mul, div, neg, pow
from .transcendental import exp, log, sqrt
from .special import norm_cdf

__all__ = [
    "add", "sub", "mul", "div", "neg", "pow",
    "exp", "log", "sqrt",
    "norm_cdf",
]
