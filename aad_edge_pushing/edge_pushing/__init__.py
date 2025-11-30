"""
Hessian computation algorithms using automatic differentiation.
From "A new framework for the computation of Hessians" by Griewank et al.

Implementations:
- algo3_block: Algorithm 3 (Block Form)
- algo4_adjmatrix: Algorithm 4 with Adjacency Matrix (O(n) neighbor lookup)
- algo4_adjlist: Algorithm 4 with Adjacency List (O(degree) neighbor lookup)

Data structures:
- SymmSparseAdjMatrix: Dictionary-based symmetric sparse matrix (simple, baseline)
- SymmSparseAdjList: Adjacency list optimized symmetric sparse matrix (fast for sparse)
"""

from .algo3_block import algo3_block
from .algo4_adjmatrix import algo4_adjmatrix
from .algo4_adjlist import algo4_adjlist
from .symm_sparse_adjmatrix import SymmSparseAdjMatrix
from .symm_sparse_adjlist import SymmSparseAdjList

__all__ = [
    'algo3_block',
    'algo4_adjmatrix',
    'algo4_adjlist',
    'SymmSparseAdjMatrix',
    'SymmSparseAdjList'
]