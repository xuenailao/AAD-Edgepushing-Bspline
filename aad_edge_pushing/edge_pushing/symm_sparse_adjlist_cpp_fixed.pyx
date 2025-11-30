# distutils: language=c++
# cython: language_level=3

"""
Fixed version of C++ sparse matrix with proper accumulation.
Simplified to avoid GIL issues.
"""

from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref, preincrement as inc

# Use int64 for composite keys
ctypedef long long int64_t
ctypedef unordered_map[int64_t, double] MapType


cdef class SymmSparseAdjListCppFixed:
    """
    Simplified C++ sparse matrix with fixed accumulation.
    """
    cdef public int n
    cdef MapType map

    def __init__(self, int n):
        self.n = n

    cdef inline int64_t _key(self, int i, int j):
        """Convert (i,j) to canonical key."""
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
        Add value to matrix element. Fixed accumulation bug.
        """
        cdef int64_t key
        cdef MapType.iterator it

        if val == 0.0:
            return

        key = self._key(i, j)
        it = self.map.find(key)

        if it != self.map.end():
            # BUG FIX: Must use map[key] assignment, not iterator modification
            self.map[key] = deref(it).second + val
            
            # Remove if cancelled out
            if abs(self.map[key]) < 1e-15:
                self.map.erase(key)
        else:
            # New entry
            self.map[key] = val

    cpdef double get(self, int i, int j):
        """Get matrix element."""
        cdef int64_t key = self._key(i, j)
        cdef MapType.iterator it = self.map.find(key)

        if it != self.map.end():
            return deref(it).second
        else:
            return 0.0

    cpdef list get_neighbors(self, int i):
        """Get all neighbors - simplified version."""
        cdef list neighbors = []
        cdef MapType.iterator it = self.map.begin()
        cdef int64_t key
        cdef int min_idx, max_idx
        cdef double val

        while it != self.map.end():
            key = deref(it).first
            val = deref(it).second
            
            # Decode key
            min_idx = <int>(key >> 32)
            max_idx = <int>(key & 0xFFFFFFFF)
            
            # Check if this entry involves node i
            if min_idx == i:
                neighbors.append((max_idx, val))
            elif max_idx == i and min_idx != max_idx:
                neighbors.append((min_idx, val))
            
            inc(it)

        return neighbors

    cpdef void clear_row_col(self, int idx):
        """Clear row and column."""
        cdef list neighbors = self.get_neighbors(idx)
        cdef int j
        cdef int64_t key

        for j, _ in neighbors:
            key = self._key(idx, j)
            self.map.erase(key)
