# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized symmetric sparse matrix with C++ STL containers.

Phase 2 optimization: Replace Python dict/set with C++ unordered_map/set
Expected speedup: Additional 2-3× over Phase 1 (47s → 15-20s)

Key improvements:
- std::unordered_map for storage (no Python object overhead)
- std::unordered_set for adjacency lists
- nogil support for future parallelization
- Custom hash function for pair<int,int>
"""

from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.vector cimport vector
from libcpp cimport bool as cbool
from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport int64_t
import numpy as np
cimport numpy as np

# Import type aliases from .pxd file
from symm_sparse_adjlist_cpp cimport MapType, AdjType


cdef class SymmSparseAdjListCpp:
    """
    C++-optimized symmetric sparse matrix with adjacency list support.

    Uses pure C++ containers for maximum performance:
    - map: unordered_map<pair<int,int>, double> with custom hash
    - adj: unordered_map<int, unordered_set<int>>

    All critical methods support nogil for parallel execution.
    """

    # Attributes declared in .pxd file
    # cdef public int n
    # cdef MapType map
    # cdef AdjType adj

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

        Returns:
            Encoded 64-bit key
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

        Args:
            i, j: Matrix indices
            val: Value to add
        """
        # All cdef declarations must be at the beginning
        cdef int64_t key
        cdef MapType.iterator it
        cdef AdjType.iterator adj_it_i
        cdef AdjType.iterator adj_it_j

        if val == 0.0:
            return

        key = self._key(i, j)
        it = self.map.find(key)

        if it != self.map.end():
            # Entry exists, accumulate
            # BUG FIX: Must update map explicitly, not through iterator
            self.map[key] = deref(it).second + val

            if abs(self.map[key]) < 1e-15:
                # Entry cancelled out, remove
                self.map.erase(it)

                # Update adjacency lists
                adj_it_i = self.adj.find(i)
                if adj_it_i != self.adj.end():
                    deref(adj_it_i).second.erase(j)
                    if deref(adj_it_i).second.empty():
                        self.adj.erase(adj_it_i)

                if i != j:
                    adj_it_j = self.adj.find(j)
                    if adj_it_j != self.adj.end():
                        deref(adj_it_j).second.erase(i)
                        if deref(adj_it_j).second.empty():
                            self.adj.erase(adj_it_j)
        else:
            # New entry
            self.map[key] = val

            # Add to adjacency list (both directions since symmetric)
            self.adj[i].insert(j)
            if i != j:
                self.adj[j].insert(i)

    cpdef double get(self, int i, int j):
        """
        Get matrix element at (i,j). Returns 0 if not stored.

        Args:
            i, j: Matrix indices

        Returns:
            Matrix element value
        """
        cdef int64_t key = self._key(i, j)
        cdef MapType.iterator it = self.map.find(key)

        if it != self.map.end():
            return deref(it).second
        else:
            return 0.0

    cpdef void set_val(self, int i, int j, double val):
        """
        Set matrix element at (i,j) to specific value (overwrites existing).

        Args:
            i, j: Matrix indices
            val: Value to set
        """
        # All cdef declarations at the beginning
        cdef int64_t key
        cdef MapType.iterator it
        cdef AdjType.iterator adj_it_i
        cdef AdjType.iterator adj_it_j

        key = self._key(i, j)

        if val == 0.0 or abs(val) < 1e-15:
            # Remove entry if exists
            it = self.map.find(key)
            if it != self.map.end():
                self.map.erase(it)

                # Update adjacency lists
                adj_it_i = self.adj.find(i)
                if adj_it_i != self.adj.end():
                    deref(adj_it_i).second.erase(j)
                    if deref(adj_it_i).second.empty():
                        self.adj.erase(adj_it_i)

                if i != j:
                    adj_it_j = self.adj.find(j)
                    if adj_it_j != self.adj.end():
                        deref(adj_it_j).second.erase(i)
                        if deref(adj_it_j).second.empty():
                            self.adj.erase(adj_it_j)
        else:
            # Set entry
            self.map[key] = val
            self.adj[i].insert(j)
            if i != j:
                self.adj[j].insert(i)

    cpdef list get_neighbors(self, int i):
        """
        Get all neighbors of node i (i.e., all j where W(i,j) ≠ 0).

        KEY OPTIMIZATION: This is O(degree(i)) instead of O(n)!

        Args:
            i: Node index

        Returns:
            List of (j, W(i,j)) pairs for all non-zero W(i,j)
        """
        # All cdef declarations at the beginning
        cdef list neighbors = []
        cdef AdjType.iterator adj_it
        cdef unordered_set[int].iterator it
        cdef int j
        cdef double val

        adj_it = self.adj.find(i)
        if adj_it == self.adj.end():
            return neighbors

        it = deref(adj_it).second.begin()
        while it != deref(adj_it).second.end():
            j = deref(it)
            val = self.get(i, j)
            if val != 0.0:
                neighbors.append((j, val))
            inc(it)

        return neighbors

    cpdef void clear_row_col(self, int idx):
        """
        Clear row and column idx from the matrix.

        This is needed in Algorithm 4 after processing a node.
        OPTIMIZED: Uses adjacency list to find entries to remove.

        Args:
            idx: Row/column index to clear
        """
        # All cdef declarations at the beginning
        cdef AdjType.iterator adj_it
        cdef vector[int] neighbors_to_clear
        cdef unordered_set[int].iterator it
        cdef int j
        cdef int64_t key
        cdef MapType.iterator map_it
        cdef AdjType.iterator adj_it_j

        adj_it = self.adj.find(idx)
        if adj_it == self.adj.end():
            return  # No entries involving idx

        # Get all neighbors of idx
        it = deref(adj_it).second.begin()
        while it != deref(adj_it).second.end():
            neighbors_to_clear.push_back(deref(it))
            inc(it)

        # Remove all entries (idx, j) and update adjacency
        for j in neighbors_to_clear:
            key = self._key(idx, j)
            map_it = self.map.find(key)
            if map_it != self.map.end():
                self.map.erase(map_it)

            # Remove from neighbor's adjacency list
            adj_it_j = self.adj.find(j)
            if adj_it_j != self.adj.end():
                deref(adj_it_j).second.erase(idx)
                if deref(adj_it_j).second.empty():
                    self.adj.erase(adj_it_j)

        # Remove idx from adjacency list
        self.adj.erase(idx)

    cpdef int nnz(self):
        """
        Return number of non-zero entries (counting symmetric pairs).

        Returns:
            Number of non-zero elements in full symmetric matrix
        """
        # All cdef declarations at the beginning
        cdef int count = 0
        cdef MapType.iterator it
        cdef int64_t key
        cdef int i, j

        it = self.map.begin()
        while it != self.map.end():
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
        """
        Return sparsity as percentage.

        Returns:
            Sparsity percentage (0-100)
        """
        return 100.0 * (1.0 - <double>self.nnz() / <double>(self.n * self.n))

    def to_dense(self):
        """
        Convert to dense numpy array (for testing/debugging).

        Returns:
            Dense n x n symmetric numpy array
        """
        # All cdef declarations at the beginning
        cdef np.ndarray[np.float64_t, ndim=2] dense
        cdef MapType.iterator it
        cdef int64_t key
        cdef double val
        cdef int i, j

        dense = np.zeros((self.n, self.n), dtype=np.float64)
        it = self.map.begin()

        while it != self.map.end():
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
        """
        Iterator over non-zero entries as ((i,j), value) tuples.
        Only returns upper triangular entries.

        Yields:
            Tuples of ((i, j), value)
        """
        # All cdef declarations at the beginning
        cdef MapType.iterator it
        cdef int64_t key
        cdef double val
        cdef int i, j

        it = self.map.begin()
        while it != self.map.end():
            key = deref(it).first
            val = deref(it).second
            # Decode key
            i = <int>(key >> 32)
            j = <int>(key & 0xFFFFFFFF)
            yield ((i, j), val)
            inc(it)

    cpdef void clear(self):
        """Clear all entries from the matrix."""
        self.map.clear()
        self.adj.clear()

    # ========================================================================
    # nogil versions for OpenMP parallelization (OpenMP v3)
    # ========================================================================

    cdef void add_nogil(self, int i, int j, double val) nogil:
        """
        Add value to matrix element (i,j) - nogil version for OpenMP.

        This version supports nogil for true parallelization with OpenMP.
        Note: NOT thread-safe! Caller must use synchronization (critical section).

        Args:
            i, j: Matrix indices
            val: Value to add
        """
        cdef int64_t key
        cdef MapType.iterator it

        if val == 0.0:
            return

        key = self._key(i, j)
        it = self.map.find(key)

        if it != self.map.end():
            # Entry exists, accumulate
            self.map[key] = deref(it).second + val
        else:
            # New entry
            self.map[key] = val
            # Note: adjacency list updates removed for nogil compatibility
            # Caller should use get_neighbors() separately if needed

    cdef double get_nogil(self, int i, int j) nogil:
        """
        Get matrix element at (i,j) - nogil version for OpenMP.

        Args:
            i, j: Matrix indices

        Returns:
            Matrix element value (0.0 if not stored)
        """
        cdef int64_t key = self._key(i, j)
        cdef MapType.iterator it = self.map.find(key)

        if it != self.map.end():
            return deref(it).second
        else:
            return 0.0
