"""
Hessian computation algorithms using automatic differentiation.
From "A new framework for the computation of Hessians" by Griewank et al.

Implementations:
- algo4_adjlist: Algorithm 4 with Adjacency List (O(degree) neighbor lookup)

Data structures:
- SymmSparseAdjMatrix: Dictionary-based symmetric sparse matrix (simple, baseline)
- SymmSparseAdjList: Adjacency list optimized symmetric sparse matrix (fast for sparse)
"""

from .algo4_adjlist import algo4_adjlist
from .symm_sparse_adjmatrix import SymmSparseAdjMatrix
from .symm_sparse_adjlist import SymmSparseAdjList

__all__ = [
    'algo4_adjlist',
    'SymmSparseAdjMatrix',
    'SymmSparseAdjList'
]