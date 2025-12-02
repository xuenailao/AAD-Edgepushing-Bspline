"""
Sparse Edge-Pushing Algorithm 4 - Optimized for PDE graphs.

Key optimization: Only track W entries related to input parameters.

In standard algo4:
- W matrix tracks Hessian contributions between ALL nodes
- For n_nodes=11K, this becomes very expensive

In sparse algo4:
- Only track W[i,j] where i OR j is an input parameter (or their direct descendants)
- Skip nodes that don't contribute to input Hessian
- Much faster for PDE graphs where inputs are heavily referenced

Complexity reduction:
- Standard: O(n_nodes × avg_degree²)
- Sparse: O(n_relevant × avg_degree²) where n_relevant << n_nodes
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict


class SparseWMatrix:
    """
    Sparse W matrix that only tracks entries relevant to input parameters.

    Instead of storing W[i,j] for all i,j, we only store:
    1. W[input_i, input_j] - final Hessian entries
    2. W[i, input_j] where i is on path from inputs to output
    """

    def __init__(self, input_indices: Set[int]):
        self.input_indices = input_indices
        # W[i][j] = value, stored as nested dict for sparsity
        self._data: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._neighbors: Dict[int, Set[int]] = defaultdict(set)

    def add(self, i: int, j: int, value: float):
        """Add value to W[i,j] and W[j,i] (symmetric)."""
        if value == 0.0:
            return

        # Ensure i <= j for canonical ordering
        if i > j:
            i, j = j, i

        self._data[i][j] += value
        self._neighbors[i].add(j)
        if i != j:
            self._neighbors[j].add(i)

    def get(self, i: int, j: int) -> float:
        """Get W[i,j]."""
        if i > j:
            i, j = j, i
        return self._data[i].get(j, 0.0)

    def get_neighbors(self, i: int) -> List[Tuple[int, float]]:
        """Get all (j, W[i,j]) pairs for node i."""
        neighbors = []
        for j in self._neighbors.get(i, set()):
            if i <= j:
                val = self._data[i].get(j, 0.0)
            else:
                val = self._data[j].get(i, 0.0)
            if val != 0.0:
                neighbors.append((j, val))
        return neighbors

    def clear_row_col(self, i: int):
        """Clear row and column i."""
        # Clear entries where i is the first index
        if i in self._data:
            for j in list(self._data[i].keys()):
                self._neighbors[j].discard(i)
            del self._data[i]

        # Clear entries where i is the second index
        for j in list(self._neighbors.get(i, set())):
            if j in self._data and i in self._data[j]:
                del self._data[j][i]

        if i in self._neighbors:
            del self._neighbors[i]


def sort_inputs_rowmajor(inputs: List) -> List:
    """
    Sort inputs to row-major order based on coefficient names.

    Assumes inputs have names like 'w0,3' (coefficient at row 0, col 3).
    Returns inputs sorted by (row, col) indices.

    This ensures Hessian ordering matches Bumping2's row-major iteration.

    Args:
        inputs: List of input ADVars with names like 'w0,3', 'w1,2', etc.

    Returns:
        Sorted list of input ADVars in row-major order.
    """
    def parse_name(name):
        # 'w0,3' -> (0, 3)
        if name and name.startswith('w') and ',' in name:
            parts = name[1:].split(',')
            return int(parts[0]), int(parts[1])
        return (float('inf'), float('inf'))  # Unknown names sort last

    inputs_with_idx = [(inp, parse_name(getattr(inp, 'name', ''))) for inp in inputs]
    inputs_sorted = sorted(inputs_with_idx, key=lambda x: (x[1][0], x[1][1]))
    return [inp for inp, _ in inputs_sorted]


def algo4_sparse(output, inputs: List, sort_inputs: bool = False) -> np.ndarray:
    """
    Sparse Edge-Pushing Algorithm 4.

    Optimized for graphs where input parameters are heavily referenced.
    Only tracks W entries relevant to final Hessian.

    Args:
        output: Output ADVar (scalar function output)
        inputs: List of input ADVars
        sort_inputs: If True, sort inputs to row-major order before computing.
                     Use this when comparing with Bumping2 which iterates row-major.

    Returns:
        Dense Hessian matrix of shape (n_inputs, n_inputs)
    """
    from ..aad.core.tape import global_tape

    # Optionally sort inputs to row-major order
    if sort_inputs:
        inputs = sort_inputs_rowmajor(inputs)

    # Create index mapping
    mapping = _create_index_mapping(global_tape, inputs, output)
    var_to_idx = mapping['var_to_idx']
    input_indices = set(mapping['input_indices'])
    output_idx = mapping['output_idx']
    n_nodes = mapping['n_nodes']

    # Pre-compute which nodes are "relevant" (can affect input Hessian)
    relevant_nodes = _find_relevant_nodes(global_tape.nodes, var_to_idx, input_indices)

    # Initialize sparse W matrix and adjoint vector
    W = SparseWMatrix(input_indices)
    vbar = [0.0] * n_nodes
    vbar[output_idx] = 1.0

    # Reverse sweep with sparse tracking
    _reverse_sweep_sparse(global_tape.nodes, W, vbar, var_to_idx,
                          input_indices, relevant_nodes)

    # Extract input Hessian
    return _extract_input_hessian(W, mapping['input_indices'])


def _find_relevant_nodes(nodes, var_to_idx: Dict, input_indices: Set[int]) -> Set[int]:
    """
    Find nodes that are relevant for input Hessian computation.

    A node is relevant if it lies on a path from an input to the output.
    We use forward reachability from inputs.
    """
    relevant = set(input_indices)

    # Forward pass: mark nodes reachable from inputs
    for node in nodes:
        i = var_to_idx.get(id(node.out), -1)
        if i < 0:
            continue

        # Check if any parent is relevant
        for parent, _ in node.parents:
            parent_idx = var_to_idx.get(id(parent), -1)
            if parent_idx in relevant:
                relevant.add(i)
                break

    return relevant


def _reverse_sweep_sparse(nodes, W: SparseWMatrix, vbar: List[float],
                          var_to_idx: Dict, input_indices: Set[int],
                          relevant_nodes: Set[int]):
    """
    Sparse reverse sweep.

    Only process nodes that are relevant for input Hessian.
    """
    n_nodes = len(nodes)

    for node_idx in range(n_nodes - 1, -1, -1):
        node = nodes[node_idx]
        i = var_to_idx.get(id(node.out), -1)
        if i < 0:
            continue

        # Skip if this node is not relevant
        if i not in relevant_nodes:
            continue

        # Get predecessors and derivatives
        preds = _get_predecessors(node, var_to_idx)
        preds_unique = sorted(set(preds))
        d1 = _get_first_derivatives(node, var_to_idx)
        d2 = _compute_second_derivatives(node, var_to_idx)

        # Check if any predecessor is an input or relevant
        has_relevant_pred = any(p in relevant_nodes for p in preds_unique)

        if not has_relevant_pred:
            continue

        # PUSHING STAGE (only for relevant paths)
        _pushing_stage_sparse(W, i, preds_unique, d1, input_indices, relevant_nodes)

        # CREATING STAGE
        vbar_i = vbar[i]
        if vbar_i != 0:
            _creating_stage_sparse(W, preds_unique, d2, vbar_i, input_indices, relevant_nodes)

            # ADJOINT STAGE
            for j in preds_unique:
                dj = d1.get(j, 0.0)
                if dj != 0.0:
                    vbar[j] += vbar_i * dj


def _pushing_stage_sparse(W: SparseWMatrix, i: int, preds: List[int],
                          d1: Dict[int, float], input_indices: Set[int],
                          relevant_nodes: Set[int]):
    """
    Sparse pushing stage.

    Only update W entries that involve input parameters or relevant nodes.
    """
    neighbors = W.get_neighbors(i)
    n_preds = len(preds)

    for p, w_pi in neighbors:
        # Skip if p is not relevant
        if p not in relevant_nodes and p not in input_indices:
            continue

        if p == i:
            # Diagonal case: W(i,i) contribution
            for a in range(n_preds):
                j = preds[a]
                # Skip if j is not relevant
                if j not in relevant_nodes and j not in input_indices:
                    continue

                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue

                for b in range(a, n_preds):
                    k = preds[b]
                    if k not in relevant_nodes and k not in input_indices:
                        continue

                    dk = d1.get(k, 0.0)
                    if dk != 0.0:
                        W.add(j, k, dj * dk * w_pi)
        else:
            # Off-diagonal case: W(p,i) contribution
            for j in preds:
                if j not in relevant_nodes and j not in input_indices:
                    continue

                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue

                if j == p:
                    W.add(p, p, 2.0 * dj * w_pi)
                else:
                    W.add(p, j, dj * w_pi)

    # Clear row/column i
    W.clear_row_col(i)


def _creating_stage_sparse(W: SparseWMatrix, preds: List[int],
                           d2: Dict[Tuple[int, int], float], vbar: float,
                           input_indices: Set[int], relevant_nodes: Set[int]):
    """Sparse creating stage."""
    if vbar == 0.0:
        return

    for (j, k), val in d2.items():
        if k < j:
            continue

        # Only add if both j and k are relevant
        j_relevant = j in relevant_nodes or j in input_indices
        k_relevant = k in relevant_nodes or k in input_indices

        if j_relevant and k_relevant and val != 0.0:
            W.add(j, k, vbar * val)


def _create_index_mapping(tape, inputs: List, output) -> Dict[str, Any]:
    """Create bidirectional mapping between ADVar objects and integer indices."""
    var_to_idx = {}
    var_to_var = {}
    idx = 0

    # Map inputs first
    input_indices = []
    for inp in inputs:
        var_to_idx[id(inp)] = idx
        var_to_var[idx] = inp
        input_indices.append(idx)
        idx += 1

    # Map output and all intermediate nodes
    for node in tape.nodes:
        if id(node.out) not in var_to_idx:
            var_to_idx[id(node.out)] = idx
            var_to_var[idx] = node.out
            idx += 1
        for parent, _ in node.parents:
            if id(parent) not in var_to_idx:
                var_to_idx[id(parent)] = idx
                var_to_var[idx] = parent
                idx += 1

    if id(output) not in var_to_idx:
        var_to_idx[id(output)] = idx
        var_to_var[idx] = output
        idx += 1

    return {
        'var_to_idx': var_to_idx,
        'idx_to_var': var_to_var,
        'input_indices': input_indices,
        'output_idx': var_to_idx[id(output)],
        'n_nodes': idx
    }


def _get_predecessors(node, var_to_idx: Dict[int, int]) -> List[int]:
    """Get predecessor indices for a given node."""
    return [var_to_idx[id(parent)] for parent, _ in node.parents]


def _get_first_derivatives(node, var_to_idx: Dict[int, int]) -> Dict[int, float]:
    """Extract first-order local derivatives, summing repeated variables."""
    d1 = {}
    for parent, deriv in node.parents:
        j = var_to_idx[id(parent)]
        d1[j] = d1.get(j, 0.0) + float(deriv)
    return d1


def _compute_second_derivatives(node, var_to_idx: Dict[int, int]) -> Dict[Tuple[int, int], float]:
    """Compute second-order local derivatives for a node."""
    d2 = {}

    if node.op_tag == 'mul' and len(node.parents) == 2:
        parent0, parent1 = node.parents[0][0], node.parents[1][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        idx1 = var_to_idx.get(id(parent1), -1)

        if idx0 >= 0 and idx1 >= 0:
            if idx0 == idx1:
                d2[(idx0, idx0)] = 2.0
            else:
                d2[(min(idx0, idx1), max(idx0, idx1))] = 1.0

    elif node.op_tag in ('add', 'sub'):
        pass

    elif node.op_tag == 'div' and len(node.parents) == 2:
        parent0, parent1 = node.parents[0][0], node.parents[1][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        idx1 = var_to_idx.get(id(parent1), -1)

        if idx0 >= 0 and idx1 >= 0:
            y_val = getattr(parent1, 'val', 1.0)
            if y_val != 0:
                if idx0 != idx1:
                    d2[(min(idx0, idx1), max(idx0, idx1))] = -1.0 / (y_val * y_val)
                x_val = getattr(parent0, 'val', 1.0)
                d2[(idx1, idx1)] = 2.0 * x_val / (y_val ** 3)

    elif node.op_tag == 'pow' and len(node.parents) == 2:
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 1.0)
            n_val = getattr(node.parents[1][0], 'val', 2.0)
            if x_val != 0:
                d2[(idx0, idx0)] = n_val * (n_val - 1) * (x_val ** (n_val - 2))

    elif node.op_tag == 'exp' and len(node.parents) == 1:
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 0.0)
            d2[(idx0, idx0)] = np.exp(x_val)

    elif node.op_tag == 'log' and len(node.parents) == 1:
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 1.0)
            if x_val > 0:
                d2[(idx0, idx0)] = -1.0 / (x_val * x_val)

    elif node.op_tag == 'erf' and len(node.parents) == 1:
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 0.0)
            d2[(idx0, idx0)] = -(4.0 * x_val / np.sqrt(np.pi)) * np.exp(-x_val * x_val)

    return d2


def _extract_input_hessian(W: SparseWMatrix, input_indices: List[int]) -> np.ndarray:
    """Extract submatrix of W corresponding to input variables."""
    n = len(input_indices)
    H = np.zeros((n, n))

    for i, idx_i in enumerate(input_indices):
        for j, idx_j in enumerate(input_indices):
            if j < i:
                continue
            H[i, j] = W.get(idx_i, idx_j)
            if i != j:
                H[j, i] = H[i, j]

    return H
