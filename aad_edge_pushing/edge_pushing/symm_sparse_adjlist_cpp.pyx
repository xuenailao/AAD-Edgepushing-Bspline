# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized symmetric sparse matrix with C++ STL containers.

Key improvements:
- std::unordered_map for storage (no Python object overhead)
- std::unordered_set for adjacency lists
- Custom int64 key encoding for (i,j) pairs
"""

from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp cimport bool as cbool
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
from libc.math cimport fabs
import numpy as np
cimport numpy as np

# Type aliases - defined inline
ctypedef unordered_map[int64_t, double] MapType
ctypedef unordered_map[int, unordered_set[int]] AdjType


cdef class SymmSparseAdjListCpp:
    """
    C++-optimized symmetric sparse matrix with adjacency list support.

    Uses pure C++ containers for maximum performance:
    - map: unordered_map<int64_t, double> (key encodes (i,j) pair)
    - adj: unordered_map<int, unordered_set<int>>
    """

    cdef public int n
    cdef MapType _map
    cdef AdjType _adj

    def __init__(self, int n):
        """
        Initialize symmetric sparse matrix of dimension n x n.

        Args:
            n: Matrix dimension
        """
        self.n = n
        # C++ containers initialized automatically

    cdef inline int64_t _key(self, int i, int j) nogil:
        """
        Convert (i,j) to canonical key for upper triangular storage.
        Encodes (min,max) as int64: (min << 32) | max
        """
        cdef int min_idx, max_idx
        if i <= j:
            min_idx = i
            max_idx = j
        else:
            min_idx = j
            max_idx = i
        return (<int64_t>min_idx << 32) | <int64_t>max_idx

    cpdef void add(self, int i, int j, double val):
        """
        Add value to matrix element at (i,j). Accumulates if entry exists.
        """
        cdef int64_t key
        cdef double current_val, new_val

        if val == 0.0:
            return

        key = self._key(i, j)

        if self._map.count(key) > 0:
            # Entry exists, accumulate
            current_val = self._map[key]
            new_val = current_val + val

            if fabs(new_val) < 1e-15:
                # Entry cancelled out, remove
                self._map.erase(key)
                # Update adjacency lists
                if self._adj.count(i) > 0:
                    self._adj[i].erase(j)
                    if self._adj[i].empty():
                        self._adj.erase(i)
                if i != j and self._adj.count(j) > 0:
                    self._adj[j].erase(i)
                    if self._adj[j].empty():
                        self._adj.erase(j)
            else:
                self._map[key] = new_val
        else:
            # New entry
            self._map[key] = val
            # Add to adjacency list (both directions since symmetric)
            self._adj[i].insert(j)
            if i != j:
                self._adj[j].insert(i)

    cpdef double get(self, int i, int j):
        """
        Get matrix element at (i,j). Returns 0 if not stored.
        """
        cdef int64_t key = self._key(i, j)
        if self._map.count(key) > 0:
            return self._map[key]
        return 0.0

    cpdef void set_val(self, int i, int j, double val):
        """
        Set matrix element at (i,j) to specific value (overwrites existing).
        """
        cdef int64_t key = self._key(i, j)

        if val == 0.0 or fabs(val) < 1e-15:
            # Remove entry if exists
            if self._map.count(key) > 0:
                self._map.erase(key)
                # Update adjacency lists
                if self._adj.count(i) > 0:
                    self._adj[i].erase(j)
                    if self._adj[i].empty():
                        self._adj.erase(i)
                if i != j and self._adj.count(j) > 0:
                    self._adj[j].erase(i)
                    if self._adj[j].empty():
                        self._adj.erase(j)
        else:
            # Set entry
            self._map[key] = val
            self._adj[i].insert(j)
            if i != j:
                self._adj[j].insert(i)

    cpdef list get_neighbors(self, int i):
        """
        Get all neighbors of node i (i.e., all j where W(i,j) ≠ 0).
        KEY OPTIMIZATION: This is O(degree(i)) instead of O(n)!
        """
        cdef list neighbors = []
        cdef int j
        cdef double val
        cdef vector[int] neighbor_vec
        cdef unordered_set[int].iterator it

        if self._adj.count(i) == 0:
            return neighbors

        # Copy to vector first
        it = self._adj[i].begin()
        while it != self._adj[i].end():
            neighbor_vec.push_back(deref(it))
            inc(it)

        # Build result list
        for j in neighbor_vec:
            val = self.get(i, j)
            if val != 0.0:
                neighbors.append((j, val))

        return neighbors

    cpdef void clear_row_col(self, int idx):
        """
        Clear row and column idx from the matrix.
        OPTIMIZED: Uses adjacency list to find entries to remove.
        """
        cdef vector[int] neighbors_to_clear
        cdef int j
        cdef int64_t key
        cdef unordered_set[int].iterator it

        if self._adj.count(idx) == 0:
            return  # No entries involving idx

        # Get all neighbors of idx
        it = self._adj[idx].begin()
        while it != self._adj[idx].end():
            neighbors_to_clear.push_back(deref(it))
            inc(it)

        # Remove all entries (idx, j) and update adjacency
        for j in neighbors_to_clear:
            key = self._key(idx, j)
            self._map.erase(key)

            # Remove from neighbor's adjacency list
            if self._adj.count(j) > 0:
                self._adj[j].erase(idx)
                if self._adj[j].empty():
                    self._adj.erase(j)

        # Remove idx from adjacency list
        self._adj.erase(idx)

    cpdef int nnz(self):
        """
        Return number of non-zero entries (counting symmetric pairs).
        """
        cdef int count = 0
        cdef int64_t key
        cdef int i, j
        cdef MapType.iterator it

        it = self._map.begin()
        while it != self._map.end():
            key = deref(it).first
            # Decode key: i = key >> 32, j = key & 0xFFFFFFFF
            i = <int>(key >> 32)
            j = <int>(key & 0xFFFFFFFF)
            if i == j:
                count += 1  # Diagonal
            else:
                count += 2  # Off-diagonal (symmetric)
            inc(it)

        return count

    cpdef double sparsity(self):
        """Return sparsity as percentage."""
        return 100.0 * (1.0 - <double>self.nnz() / <double>(self.n * self.n))

    def to_dense(self):
        """Convert to dense numpy array (for testing/debugging)."""
        cdef np.ndarray[np.float64_t, ndim=2] dense
        cdef int64_t key
        cdef double val
        cdef int i, j
        cdef MapType.iterator it

        dense = np.zeros((self.n, self.n), dtype=np.float64)
        it = self._map.begin()

        while it != self._map.end():
            key = deref(it).first
            val = deref(it).second
            # Decode key
            i = <int>(key >> 32)
            j = <int>(key & 0xFFFFFFFF)

            dense[i, j] = val
            if i != j:
                dense[j, i] = val

            inc(it)

        return dense

    def items(self):
        """Iterator over non-zero entries as ((i,j), value) tuples."""
        cdef int64_t key
        cdef double val
        cdef int i, j
        cdef MapType.iterator it

        it = self._map.begin()
        while it != self._map.end():
            key = deref(it).first
            val = deref(it).second
            # Decode key
            i = <int>(key >> 32)
            j = <int>(key & 0xFFFFFFFF)
            yield ((i, j), val)
            inc(it)

    cpdef void clear(self):
        """Clear all entries from the matrix."""
        self._map.clear()
        self._adj.clear()

    # ========================================================================
    # nogil versions for OpenMP parallelization
    # ========================================================================

    cdef void add_nogil(self, int i, int j, double val) nogil:
        """
        Add value to matrix element (i,j) - nogil version.
        Note: NOT thread-safe! Caller must use synchronization.
        """
        cdef int64_t key
        cdef double current_val

        if val == 0.0:
            return

        key = self._key(i, j)

        if self._map.count(key) > 0:
            current_val = self._map[key]
            self._map[key] = current_val + val
        else:
            self._map[key] = val

    cdef double get_nogil(self, int i, int j) nogil:
        """Get matrix element at (i,j) - nogil version."""
        cdef int64_t key = self._key(i, j)
        if self._map.count(key) > 0:
            return self._map[key]
        return 0.0
