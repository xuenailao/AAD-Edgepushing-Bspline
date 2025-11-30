"""
Optimized symmetric sparse matrix with adjacency list support.

Implementation: Adjacency List
- Maintains adjacency lists to find neighbors of a node in O(degree) instead of O(n)
- Critical for sparse Hessian computation where nnz << n²

Performance gain:
- Old (AdjMatrix): O(n) scan to find neighbors → 67% of Algo4 time
- New (AdjList): O(degree) lookup → Proportional to actual sparsity
"""

import numpy as np
from typing import Dict, Tuple, List, Set, Iterator
from collections import defaultdict


class SymmSparseAdjList:
    """
    Symmetric sparse matrix with O(1) neighbor lookup.

    Maintains two data structures:
    1. map: Dict[(i,j), val] - Canonical storage (upper triangular)
    2. adj: Dict[i, Set[j]] - Adjacency list for fast neighbor queries

    This allows Algorithm 4 to find non-zero W(p,i) in O(degree(i))
    instead of O(n).
    """

    def __init__(self, n: int):
        """
        Initialize symmetric sparse matrix of dimension n x n.

        Args:
            n: Matrix dimension
        """
        self.n = n
        self.map: Dict[Tuple[int, int], float] = {}
        # Adjacency list: adj[i] = set of all j where W(i,j) ≠ 0 or W(j,i) ≠ 0
        self.adj: Dict[int, Set[int]] = defaultdict(set)

    def _key(self, i: int, j: int) -> Tuple[int, int]:
        """
        Convert (i,j) to canonical key (min,max) for upper triangular storage.
        """
        return (i, j) if i <= j else (j, i)

    def add(self, i: int, j: int, val: float) -> None:
        """
        Add value to matrix element at (i,j). Accumulates if entry exists.
        Automatically maintains adjacency list.

        Args:
            i, j: Row and column indices
            val: Value to add (skips if val == 0)
        """
        if val == 0:
            return

        key = self._key(i, j)

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
            self.adj[i].add(j)
            if i != j:
                self.adj[j].add(i)

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
        key = self._key(i, j)

        if val == 0 or abs(val) < 1e-15:
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
            self.adj[i].add(j)
            if i != j:
                self.adj[j].add(i)

    def get_neighbors(self, i: int) -> List[Tuple[int, float]]:
        """
        Get all neighbors of node i (i.e., all j where W(i,j) ≠ 0).

        KEY OPTIMIZATION: This is O(degree(i)) instead of O(n)!

        Args:
            i: Node index

        Returns:
            List of (j, W(i,j)) pairs for all non-zero W(i,j)
        """
        neighbors = []
        if i in self.adj:
            for j in self.adj[i]:
                val = self.get(i, j)
                if val != 0.0:
                    neighbors.append((j, val))
        return neighbors

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
        self.adj.clear()

    def clear_row_col(self, idx: int) -> None:
        """
        Clear row and column idx from the matrix.
        This is needed in Algorithm 3 after processing a node.

        OPTIMIZED: Uses adjacency list to find entries to remove.

        Args:
            idx: Row/column index to clear
        """
        if idx not in self.adj:
            return  # No entries involving idx

        # Get all neighbors of idx
        neighbors_to_clear = list(self.adj[idx])

        # Remove all entries (idx, j) and update adjacency
        for j in neighbors_to_clear:
            key = self._key(idx, j)
            if key in self.map:
                del self.map[key]

            # Remove from neighbor's adjacency list
            self.adj[j].discard(idx)
            if not self.adj[j]:
                del self.adj[j]

        # Remove idx from adjacency list
        del self.adj[idx]

    def nnz(self) -> int:
        """Return number of non-zero entries (counting symmetric pairs)."""
        count = 0
        for (i, j) in self.map.keys():
            count += 1 if i == j else 2  # Diagonal or off-diagonal
        return count

    def sparsity(self) -> float:
        """Return sparsity as percentage."""
        return 100.0 * (1.0 - self.nnz() / (self.n * self.n))
