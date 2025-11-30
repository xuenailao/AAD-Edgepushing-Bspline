# aad/core/tape.py
from __future__ import annotations
from typing import List, Tuple, Iterable, Optional
from contextlib import contextmanager
from .node import Node

class Tape:
    """
    A simple global tape: records Nodes in forward order.
    """
    def __init__(self):
        self.nodes: List[Node] = []

    def reset(self):
        self.nodes.clear()

    def push_node(self, *, op_tag: str, out, parents: List[Tuple]):
        """
        Append a Node(op_tag, out, parents) to the tape.
        `parents` is a list of (parent_ADVar, local_partial_numeric).
        """
        self.nodes.append(Node(op_tag=op_tag, out=out, parents=parents))
        return len(self.nodes) - 1

# Global singleton tape (simple and practical for a first implementation)
global_tape = Tape()

@contextmanager
def use_tape(tape: Optional[Tape] = None):
    """
    Context manager to temporarily use a fresh tape:
        with use_tape():
            ... build computation ...
            reverse(y)
    """
    from . import tape as _tape_mod  # local import to avoid cycles
    prev = _tape_mod.global_tape
    try:
        _tape_mod.global_tape = tape or Tape()
        yield _tape_mod.global_tape
    finally:
        _tape_mod.global_tape = prev