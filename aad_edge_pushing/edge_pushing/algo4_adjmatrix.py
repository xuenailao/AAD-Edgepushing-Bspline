"""
Algorithm 4 (Edge-Pushing/Component-wise form) implementation.

Implementation: Adjacency Matrix (Dictionary-based)
- Uses SymmSparseAdjMatrix for W matrix storage
- O(n) neighbor lookup (must scan all n nodes)
- Simple baseline implementation

Key improvements over Algorithm 3:
1. Row-wise iteration over W matrix (instead of all {j,k} pairs)
2. Sparsity-aware: Only processes non-zero W(p,i) entries
3. Avoids redundant w_{ij} term recomputation

Performance advantage in sparse Hessian scenarios.

Reference: "A new framework for the computation of Hessians" (Griewank et al., 2008)
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from ..aad.core.tape import global_tape, Tape
from ..aad.core.var import ADVar
from ..aad.core.node import Node
from .symm_sparse_adjmatrix import SymmSparseAdjMatrix


def algo4_adjmatrix(output: ADVar, inputs: List[ADVar]) -> np.ndarray:
    """
    Compute Hessian using Algorithm 4 (Edge-Pushing/Component-wise form).

    This algorithm is more efficient than Algo3 for sparse Hessians because:
    - Only processes non-zero W(p,i) entries (sparsity-aware)
    - Row-wise iteration provides better granularity
    - Avoids visiting zero entries

    Args:
        output: Output ADVar (scalar function output)
        inputs: List of input ADVars

    Returns:
        Dense Hessian matrix of shape (n_inputs, n_inputs)
    """
    mapping = _create_index_mapping(global_tape, inputs, output)
    var_to_idx = mapping['var_to_idx']
    idx_to_var = mapping['idx_to_var']
    input_indices = mapping['input_indices']
    output_idx = mapping['output_idx']
    n_nodes = mapping['n_nodes']

    # Initialize W matrix and adjoint vector
    W = SymmSparseAdjMatrix(n_nodes)
    vbar = [0.0] * n_nodes
    vbar[output_idx] = 1.0  # Seed adjoint for output

    # Reverse sweep through nodes
    for node in reversed(global_tape.nodes):
        i = var_to_idx[id(node.out)]

        # Get predecessors and derivatives
        preds = _get_predecessors(node, var_to_idx)
        preds = sorted(set(preds))  # Remove duplicates
        d1 = _get_first_derivatives(node, var_to_idx)
        d2 = _compute_second_derivatives(node, var_to_idx)

        # PUSHING STAGE: Edge-pushing (sparsity-aware)
        _pushing_stage(W, i, preds, d1, n_nodes)

        # CREATING STAGE: Add second-order local derivatives
        if vbar[i] != 0:
            _creating_stage(W, preds, d2, vbar[i])

            # ADJOINT STAGE: v̄ᵀ ← v̄ᵀ Φ'ᵢ
            _adjoint_update(vbar, i, preds, d1)

    return _extract_input_hessian(W, input_indices)


def _pushing_stage(W: SymmSparseAdjMatrix, i: int, preds: List[int],
                   d1: Dict[int, float], n_nodes: int) -> None:
    """
    Pushing stage of Algorithm 4 (Edge-Pushing).

    Key optimization over Algo3:
    - Iterate row-wise over W to find neighbors neigh(p,i)
    - Only process non-zero W(p,i) entries (sparsity-aware)
    - Two cases:
      1. p = i: Diagonal case, W(i,i) contribution
      2. p ≠ i: Off-diagonal case, push W(p,i) to predecessors

    This avoids O(|preds|²) iteration when W is sparse.

    Args:
        W: Working matrix storing accumulated second derivatives
        i: Current node index
        preds: List of predecessor indices (deduplicated)
        d1: First-order derivatives ∂φᵢ/∂xⱼ
        n_nodes: Total number of nodes
    """
    # Collect all non-zero neighbors neigh(p,i) where W(p,i) ≠ 0
    neighbors = []

    # Extract all (p, W(p,i)) pairs where W(p,i) ≠ 0
    for p in range(n_nodes):
        w_pi = W.get(p, i)
        if w_pi != 0.0:
            neighbors.append((p, w_pi))

    # Process each neighbor
    for p, w_pi in neighbors:
        if p == i:
            # Case 1: Diagonal W(i,i) contribution
            # Formula: W(j,k) += d1[j] * d1[k] * W(i,i)
            for a, j in enumerate(preds):
                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue
                for b in range(a, len(preds)):
                    k = preds[b]
                    dk = d1.get(k, 0.0)
                    if dk != 0.0:
                        W.add(j, k, dj * dk * w_pi)
        else:
            # Case 2: Off-diagonal W(p,i) contribution
            # Push W(p,i) to W(p,j) and W(j,k) depending on whether j=p
            for j in preds:
                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue

                if j == p:
                    # Special case: j = p → diagonal update
                    # Formula: W(p,p) += 2 * d1[p] * W(p,i)
                    W.add(p, p, 2.0 * dj * w_pi)
                else:
                    # General case: j ≠ p → off-diagonal update
                    # Formula: W(p,j) += d1[j] * W(p,i)
                    W.add(p, j, dj * w_pi)

    # Clear row/column i to avoid reuse
    W.clear_row_col(i)


def _creating_stage(W: SymmSparseAdjMatrix, preds: List[int],
                    d2: Dict[Tuple[int, int], float], vbar: float) -> None:
    """
    Creating stage: W += v̄ᵢ * Φ''ᵢ

    Adds contribution from local second derivatives.
    Identical to Block 2 in Algorithm 3.

    Args:
        W: Working matrix
        preds: List of predecessor indices
        d2: Second-order derivatives ∂²φᵢ/∂xⱼ∂xₖ
        vbar: Adjoint value at current node
    """
    if vbar == 0.0:
        return  # No contribution if adjoint is zero

    for (j, k), val in d2.items():
        if k < j:
            continue  # Only fill upper triangle
        if val != 0.0:
            W.add(j, k, vbar * val)


# ============================================================================
# Shared helper functions (reused from algo3_block.py)
# ============================================================================

def _create_index_mapping(tape: Tape, inputs: List[ADVar], output: ADVar) -> Dict[str, Any]:
    """
    Create bidirectional mapping between ADVar objects and integer indices.

    Args:
        tape: The computation tape
        inputs: List of input ADVars
        output: Output ADVar

    Returns:
        Dict containing:
            - 'var_to_idx': ADVar id -> integer index
            - 'idx_to_var': integer index -> ADVar
            - 'input_indices': List of indices for input variables
            - 'output_idx': Index of output variable
            - 'n_nodes': Total number of nodes
    """
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
        # Map parents
        for parent, _ in node.parents:
            if id(parent) not in var_to_idx:
                var_to_idx[id(parent)] = idx
                var_to_var[idx] = parent
                idx += 1

    # Make sure output is in the mapping
    if id(output) not in var_to_idx:
        var_to_idx[id(output)] = idx
        var_to_var[idx] = output
        idx += 1

    output_idx = var_to_idx[id(output)]

    return {
        'var_to_idx': var_to_idx,
        'idx_to_var': var_to_var,
        'input_indices': input_indices,
        'output_idx': output_idx,
        'n_nodes': idx
    }


def _get_predecessors(node: Node, var_to_idx: Dict[int, int]) -> List[int]:
    """Get predecessor indices for a given node."""
    preds = []
    for parent, _ in node.parents:
        preds.append(var_to_idx[id(parent)])
    return preds


def _get_first_derivatives(node: Node, var_to_idx: Dict[int, int]) -> Dict[int, float]:
    """
    Extract first-order local derivatives for a node.
    Properly sums derivatives when the same variable appears multiple times.
    """
    d1 = {}
    for parent, deriv in node.parents:
        j = var_to_idx[id(parent)]
        d1[j] = d1.get(j, 0.0) + float(deriv)
    return d1


def _compute_second_derivatives(node: Node, var_to_idx: Dict[int, int]) -> Dict[Tuple[int, int], float]:
    """
    Compute second-order local derivatives for a node.
    Supports: mul, add, sub, div, pow, exp, log, erf
    """
    d2 = {}

    # Handle different operation types
    if node.op_tag == 'mul' and len(node.parents) == 2:
        parent0, parent1 = node.parents[0][0], node.parents[1][0]
        idx0, idx1 = var_to_idx.get(id(parent0), -1), var_to_idx.get(id(parent1), -1)

        if idx0 >= 0 and idx1 >= 0:
            if idx0 == idx1:  # x*x case: ∂²(x²)/∂x² = 2
                d2[(idx0, idx0)] = 2.0
            else:  # x*y case: ∂²(xy)/∂x∂y = 1
                d2[(min(idx0, idx1), max(idx0, idx1))] = 1.0

    elif node.op_tag in ('add', 'sub'):
        pass  # All second derivatives are 0

    elif node.op_tag == 'div' and len(node.parents) == 2:
        parent0, parent1 = node.parents[0][0], node.parents[1][0]
        idx0, idx1 = var_to_idx.get(id(parent0), -1), var_to_idx.get(id(parent1), -1)

        if idx0 >= 0 and idx1 >= 0:
            y_val = getattr(parent1, 'val', 1.0)
            if y_val != 0:
                if idx0 != idx1:
                    # ∂²(x/y)/∂x∂y = -1/y²
                    d2[(min(idx0, idx1), max(idx0, idx1))] = -1.0 / (y_val * y_val)

                # ∂²(x/y)/∂y² = 2x/y³
                x_val = getattr(parent0, 'val', 1.0)
                d2[(idx1, idx1)] = 2.0 * x_val / (y_val ** 3)

    elif node.op_tag == 'pow' and len(node.parents) == 2:
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 1.0)
            n_val = getattr(node.parents[1][0], 'val', 2.0)
            if x_val != 0:
                # ∂²(x^n)/∂x² = n(n-1)x^(n-2)
                d2[(idx0, idx0)] = n_val * (n_val - 1) * (x_val ** (n_val - 2))

    elif node.op_tag == 'exp' and len(node.parents) == 1:
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        if idx0 >= 0:
            # ∂²exp(x)/∂x² = exp(x)
            x_val = getattr(parent0, 'val', 0.0)
            d2[(idx0, idx0)] = np.exp(x_val)

    elif node.op_tag == 'log' and len(node.parents) == 1:
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 1.0)
            if x_val > 0:
                # ∂²log(x)/∂x² = -1/x²
                d2[(idx0, idx0)] = -1.0 / (x_val * x_val)

    elif node.op_tag == 'erf' and len(node.parents) == 1:
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 0.0)
            # ∂²erf(x)/∂x² = -(4x/√π) * exp(-x²)
            d2[(idx0, idx0)] = -(4.0 * x_val / np.sqrt(np.pi)) * np.exp(-x_val * x_val)

    return d2


def _adjoint_update(vbar: List[float], i: int, preds: List[int],
                    d1: Dict[int, float]) -> None:
    """Adjoint update: v̄ᵀ ← v̄ᵀ Φ'ᵢ"""
    if vbar[i] == 0.0:
        return
    for j in preds:
        vbar[j] += vbar[i] * d1.get(j, 0.0)


def _extract_input_hessian(W: SymmSparseAdjMatrix, input_indices: List[int]) -> np.ndarray:
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
