# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized symmetric sparse matrix - Version 2

Improved over v1 with better type declarations and optimizations
while still using Python containers (for GIL compatibility).

Future: Can be upgraded to C++ containers with proper pybind11 integration.
"""

import numpy as np
cimport numpy as np
from libc.math cimport fabs

cdef class SymmSparseAdjListV2:
    """
    Cython-optimized symmetric sparse matrix with adjacency list support.

    Version 2 improvements:
    - Better type declarations
    - Inlined critical functions
    - Reduced Python overhead
    - Optimized loops
    """

    cdef public int n
    cdef public dict map
    cdef public dict adj

    def __init__(self, int n):
        """Initialize symmetric sparse matrix of dimension n x n."""
        self.n = n
        self.map = {}
        self.adj = {}

    cdef inline tuple _key(self, int i, int j):
        """
        Convert (i,j) to canonical key (min,max) for upper triangular storage.
        Marked inline for maximum performance.
        """
        if i <= j:
            return (i, j)
        else:
            return (j, i)

    cpdef void add(self, int i, int j, double val):
        """
        Add value to matrix element at (i,j). Accumulates if entry exists.
        Automatically maintains adjacency list.
        """
        if val == 0.0:
            return

        cdef tuple key = self._key(i, j)

        if key in self.map:
            self.map[key] = self.map[key] + val
            if fabs(self.map[key]) < 1e-15:
                # Entry cancelled out, remove from map and adjacency
                del self.map[key]
                if i in self.adj and j in self.adj[i]:
                    self.adj[i].discard(j)
                    if not self.adj[i]:
                        del self.adj[i]
                if i != j:
                    if j in self.adj and i in self.adj[j]:
                        self.adj[j].discard(i)
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
        """
        cdef tuple key = self._key(i, j)
        return self.map.get(key, 0.0)

    cpdef void set_val(self, int i, int j, double val):
        """
        Set matrix element at (i,j) to specific value (overwrites existing).
        """
        cdef tuple key = self._key(i, j)

        if val == 0.0 or fabs(val) < 1e-15:
            # Remove entry if exists
            if key in self.map:
                del self.map[key]
                if i in self.adj:
                    self.adj[i].discard(j)
                    if not self.adj[i]:
                        del self.adj[i]
                if i != j and j in self.adj:
                    self.adj[j].discard(i)
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
        """
        cdef list neighbors = []
        cdef int j
        cdef double val

        if i not in self.adj:
            return neighbors

        for j in self.adj[i]:
            val = self.get(i, j)
            if val != 0.0:
                neighbors.append((j, val))

        return neighbors

    cpdef void clear_row_col(self, int idx):
        """
        Clear row and column idx from the matrix.
        This is needed in Algorithm 4 after processing a node.
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

            # Remove from neighbor's adjacency list
            if j in self.adj:
                self.adj[j].discard(idx)
                if not self.adj[j]:
                    del self.adj[j]

        # Remove idx from adjacency list
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
        """
        cdef np.ndarray[np.float64_t, ndim=2] dense = np.zeros((self.n, self.n), dtype=np.float64)
        cdef tuple key
        cdef double val
        cdef int i, j

        for key, val in self.map.items():
            i, j = key
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
