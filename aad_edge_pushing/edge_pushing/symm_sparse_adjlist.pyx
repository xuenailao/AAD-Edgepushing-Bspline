# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized symmetric sparse matrix with adjacency list support.

Phase 1 optimization: Type annotations + Cython compilation
Expected speedup: 12× (verified by 三今: 575s → 47s)

Implementation: Adjacency List
- Maintains adjacency lists to find neighbors of a node in O(degree) instead of O(n)
- Critical for sparse Hessian computation where nnz << n²
"""

import numpy as np
from typing import Tuple, List

cdef class SymmSparseAdjList:
    """
    Cython-optimized symmetric sparse matrix with O(1) neighbor lookup.

    Maintains two data structures:
    1. map: Dict[(i,j), val] - Canonical storage (upper triangular)
    2. adj: Dict[i, Set[j]] - Adjacency list for fast neighbor queries
    """

    cdef public int n
    cdef public dict map
    cdef public dict adj

    def __init__(self, int n):
        """
        Initialize symmetric sparse matrix of dimension n x n.

        Args:
            n: Matrix dimension
        """
        self.n = n
        self.map = {}
        self.adj = {}  # Use regular dict, handle defaults manually

    cdef inline tuple _key(self, int i, int j):
        """
        Convert (i,j) to canonical key (min,max) for upper triangular storage.
        Marked as cdef inline for maximum performance.
        """
        if i <= j:
            return (i, j)
        else:
            return (j, i)

    cpdef void add(self, int i, int j, double val):
        """
        Add value to matrix element at (i,j). Accumulates if entry exists.
        Automatically maintains adjacency list.

        cpdef allows calling from both Python and Cython code.
        """
        if val == 0.0:
            return

        cdef tuple key = self._key(i, j)

        if key in self.map:
            self.map[key] += val
            if abs(self.map[key]) < 1e-15:  # Treat tiny values as zero
                # Entry cancelled out, remove from map and adjacency
                del self.map[key]
                self.adj[i].discard(j)
                self.adj[j].discard(i)
                if not self.adj[i]:
                    del self.adj[i]
                if not self.adj[j]:
                    del self.adj[j]
        else:
            # New entry
            self.map[key] = val
            # Add to adjacency list (both directions since symmetric)
            if i not in self.adj:
                self.adj[i] = set()
            self.adj[i].add(j)
            if i != j:
                if j not in self.adj:
                    self.adj[j] = set()
                self.adj[j].add(i)

    cpdef double get(self, int i, int j):
        """
        Get matrix element at (i,j). Returns 0 if not stored.

        Returns:
            Matrix element value
        """
        cdef tuple key = self._key(i, j)
        return self.map.get(key, 0.0)

    cpdef void set(self, int i, int j, double val):
        """
        Set matrix element at (i,j) to specific value (overwrites existing).
        """
        cdef tuple key = self._key(i, j)

        if val == 0.0 or abs(val) < 1e-15:
            # Remove entry if exists
            if key in self.map:
                del self.map[key]
                self.adj[i].discard(j)
                self.adj[j].discard(i)
                if not self.adj[i]:
                    del self.adj[i]
                if not self.adj[j]:
                    del self.adj[j]
        else:
            # Set entry
            self.map[key] = val
            if i not in self.adj:
                self.adj[i] = set()
            self.adj[i].add(j)
            if i != j:
                if j not in self.adj:
                    self.adj[j] = set()
                self.adj[j].add(i)

    cpdef list get_neighbors(self, int i):
        """
        Get all neighbors of node i (i.e., all j where W(i,j) ≠ 0).

        KEY OPTIMIZATION: This is O(degree(i)) instead of O(n)!

        Returns:
            List of (j, W(i,j)) pairs for all non-zero W(i,j)
        """
        cdef list neighbors = []
        cdef int j
        cdef double val

        if i in self.adj:
            for j in self.adj[i]:
                val = self.get(i, j)
                if val != 0.0:
                    neighbors.append((j, val))

        return neighbors

    cpdef void clear_row_col(self, int idx):
        """
        Clear row and column idx from the matrix.
        This is needed in Algorithm 4 after processing a node.

        OPTIMIZED: Uses adjacency list to find entries to remove.
        """
        if idx not in self.adj:
            return  # No entries involving idx

        # Get all neighbors of idx
        cdef list neighbors_to_clear = list(self.adj[idx])
        cdef int j
        cdef tuple key

        # Remove all entries (idx, j) and update adjacency
        for j in neighbors_to_clear:
            key = self._key(idx, j)
            if key in self.map:
                del self.map[key]

            # Remove from neighbor's adjacency list (check if neighbor still exists)
            if j in self.adj:
                self.adj[j].discard(idx)
                if not self.adj[j]:
                    del self.adj[j]

        # Remove idx from adjacency list
        if idx in self.adj:
            del self.adj[idx]

    cpdef int nnz(self):
        """Return number of non-zero entries (counting symmetric pairs)."""
        cdef int count = 0
        cdef tuple key

        for key in self.map.keys():
            if key[0] == key[1]:
                count += 1  # Diagonal
            else:
                count += 2  # Off-diagonal (symmetric)

        return count

    cpdef double sparsity(self):
        """Return sparsity as percentage."""
        return 100.0 * (1.0 - <double>self.nnz() / <double>(self.n * self.n))

    def to_dense(self):
        """
        Convert to dense numpy array (for testing/debugging).

        Returns:
            Dense n x n symmetric matrix
        """
        cdef int i, j

        dense = np.zeros((self.n, self.n))
        for (i, j), val in self.map.items():
            dense[i, j] = val
            if i != j:
                dense[j, i] = val
        return dense

    def items(self):
        """
        Iterator over non-zero entries as ((i,j), value) pairs.
        Only returns upper triangular entries.
        """
        return iter(self.map.items())

    cpdef void clear(self):
        """Clear all entries from the matrix."""
        self.map.clear()
        self.adj.clear()
