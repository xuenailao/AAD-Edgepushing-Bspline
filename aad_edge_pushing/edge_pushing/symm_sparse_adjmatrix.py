"""
Symmetric sparse matrix data structure for efficient Hessian storage.
Only stores upper triangular part to save memory.

Implementation: Adjacency Matrix (Dictionary-based)
- Uses dict mapping (i,j) -> value
- O(n) neighbor lookup (must scan all n nodes)
- Simple implementation, good for dense matrices
"""

import numpy as np
from typing import Dict, Tuple, Iterator


class SymmSparseAdjMatrix:
    """
    Symmetric sparse matrix storing only upper triangular entries.
    Access is symmetric: get(i,j) == get(j,i)
    Designed for Algorithm 3's W matrix operations.
    """
    
    def __init__(self, n: int):
        """
        Initialize symmetric sparse matrix of dimension n x n.
        
        Args:
            n: Matrix dimension
        """
        self.n = n
        self.map: Dict[Tuple[int, int], float] = {}
        
    def _key(self, i: int, j: int) -> Tuple[int, int]:
        """
        Convert (i,j) to canonical key (min,max) for upper triangular storage.
        
        Args:
            i, j: Row and column indices
            
        Returns:
            Tuple (min(i,j), max(i,j))
        """
        return (i, j) if i <= j else (j, i)
        
    def add(self, i: int, j: int, val: float) -> None:
        """
        Add value to matrix element at (i,j). Accumulates if entry exists.
        Automatically handles symmetry.
        
        Args:
            i, j: Row and column indices
            val: Value to add (skips if val == 0)
        """
        if val == 0:
            return
        key = self._key(i, j)
        if key in self.map:
            self.map[key] += val
            if self.map[key] == 0:
                del self.map[key]
        else:
            self.map[key] = val
        
    def get(self, i: int, j: int) -> float:
        """
        Get matrix element at (i,j). Returns 0 if not stored.
        
        Args:
            i, j: Row and column indices
            
        Returns:
            Matrix element value
        """
        return self.map.get(self._key(i, j), 0.0)
        
    def set(self, i: int, j: int, val: float) -> None:
        """
        Set matrix element at (i,j) to specific value (overwrites existing).
        
        Args:
            i, j: Row and column indices
            val: Value to set
        """
        if val == 0:
            self.map.pop(self._key(i, j), None)
        else:
            self.map[self._key(i, j)] = val

    def to_dense(self) -> np.ndarray:
        """
        Convert to dense numpy array (for testing/debugging).
        
        Returns:
            Dense n x n symmetric matrix
        """
        dense = np.zeros((self.n, self.n))
        for (i, j), val in self.map.items():
            dense[i, j] = val
            if i != j:
                dense[j, i] = val
        return dense
        
    def items(self) -> Iterator[Tuple[Tuple[int, int], float]]:
        """
        Iterator over non-zero entries as ((i,j), value) pairs.
        Only returns upper triangular entries.
        """
        return iter(self.map.items())

    def clear(self) -> None:
        """
        Clear all entries from the matrix.
        """
        self.map.clear()
    
    def clear_row_col(self, idx: int) -> None:
        """
        Clear row and column idx from the matrix.
        This is needed in Algorithm 3 after processing a node.
        
        Args:
            idx: Row/column index to clear
        """
        # Create a list of keys to remove (can't modify dict during iteration)
        keys_to_remove = []
        for (i, j) in self.map.keys():
            if i == idx or j == idx:
                keys_to_remove.append((i, j))
        
        # Remove the keys
        for key in keys_to_remove:
            del self.map[key]
