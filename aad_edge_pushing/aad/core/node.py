# aad/core/node.py
from dataclasses import dataclass
from typing import List, Tuple, Any   # typing gives us generic container types for annotations

@dataclass
class Node:
    """
    One node on the tape produced by a primitive operation.

    Attributes
    ----------
    op_tag : str
        Debug tag (e.g., "add", "mul").
    out    : Any
        The ADVar produced by this op. Type is `Any` because we want flexibility:
        it could be a scalar float, a numpy array, or an ADVar wrapper.
    parents: List[Tuple[Any, Any]]
        List of (parent_var, local_partial) pairs:
          - parent_var : the ADVar input that this node depends on
          - local_partial : numeric value (float/ndarray) already shaped
            to match the output, representing ∂out/∂parent.
    """
    op_tag: str                        # operation name as a string
    out: Any                           # flexible type (usually ADVar)
    parents: List[Tuple[Any, Any]]     # list of pairs (variable, derivative)