# aad/__init__.py
# Automatic Adjoint Differentiation library

from .core.var import ADVar
from .core.tape import Tape, global_tape, use_tape
from .core.engine import (
    reverse,
    zero_adjoints,
    hvp_for,
    edge_push_hessian,
)

# Taylor expansion module
from . import taylor
from .taylor import TVar, taylor_grad_hessian, taylor_hessian

__all__ = [
    # Core
    'ADVar',
    'Tape',
    'global_tape',
    'use_tape',
    # Engine
    'reverse',
    'zero_adjoints',
    'hvp_for',
    'edge_push_hessian',
    # Taylor
    'taylor',
    'TVar',
    'taylor_grad_hessian',
    'taylor_hessian',
]
