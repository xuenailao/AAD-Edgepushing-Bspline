"""
Algorithm 3 (block form) implementation using existing AAD infrastructure.
Computes Hessian using block-wise operations on the W matrix.
"""

import numpy as np
from typing import List, Dict, Tuple, Any
from ..aad.core.tape import global_tape, Tape
from ..aad.core.var import ADVar
from ..aad.core.node import Node
# from ..aad.core.engine import _second_locals  # Not implemented yet
from .symm_sparse_adjmatrix import SymmSparseAdjMatrix


def algo3_block(output: ADVar, inputs: List[ADVar]) -> np.ndarray:
    """
    Compute Hessian using Algorithm 3 (block form).
    
    This algorithm propagates second-order information backward through 
    the computation graph using block matrix operations.
    
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
        
        # Block 1: W ← (Φ'ᵢ)ᵀ W Φ'ᵢ - ALWAYS EXECUTE (not gated by vbar)
        _block1_update(W, i, preds, d1, n_nodes)
        
        # Block 2 and Adjoint: Only execute if vbar[i] != 0
        if vbar[i] != 0:
            # Block 2: W += v̄ᵢ * Φ''ᵢ
            _block2_update(W, preds, d2, vbar[i])
            
            # Adjoint: v̄ᵀ ← v̄ᵀ Φ'ᵢ
            _adjoint_update(vbar, i, preds, d1)
    
    return _extract_input_hessian(W, input_indices)

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
    
    # Map output
    for nodes in tape.nodes:
        if id(nodes.out) not in var_to_idx:
            var_to_idx[id(nodes.out)] = idx
            var_to_var[idx] = nodes.out
            idx += 1
        # Map parents
        for parent,_ in nodes.parents:
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
        'n_nodes': idx}

def _get_predecessors(node: Node, var_to_idx: Dict[int, int]) -> List[int]:
    """
    Get predecessor indices for a given node.
    
    Args:
        node: Current node
        var_to_idx: Mapping from ADVar id to index
        
    Returns:
        List of predecessor node indices
    """
    preds = []
    for parent, _ in node.parents:
        preds.append(var_to_idx[id(parent)])
    
    return preds


def _get_first_derivatives(node: Node, var_to_idx: Dict[int, int]) -> Dict[int, float]:
    """
    Extract first-order local derivatives for a node.
    
    FIXED: Now properly sums derivatives when the same variable appears
    multiple times as a parent (e.g., x*x).
    
    Args:
        node: Current node
        var_to_idx: Mapping from ADVar id to index
        
    Returns:
        Dict mapping predecessor index j to ∂φᵢ/∂xⱼ (summed if repeated)
    """
    d1 = {}
    for parent, deriv in node.parents:
        j = var_to_idx[id(parent)]
        # Sum derivatives for repeated variables
        d1[j] = d1.get(j, 0.0) + float(deriv)
    return d1


def _compute_second_derivatives(node: Node, var_to_idx: Dict[int, int]) -> Dict[Tuple[int, int], float]:
    """
    Compute second-order local derivatives for a node.
    
    Supports common operations: mul, add, sub, div, pow, exp, log
    
    Args:
        node: Current node
        var_to_idx: Mapping from ADVar id to index
        
    Returns:
        Dict mapping (j,k) to ∂²φᵢ/∂xⱼ∂xₖ
    """
    d2 = {}
    
    # Handle different operation types
    if node.op_tag == 'mul' and len(node.parents) == 2:
        # Multiplication: f = x * y
        parent0 = node.parents[0][0]
        parent1 = node.parents[1][0]
        idx0 = var_to_idx.get(id(parent0), -1)
        idx1 = var_to_idx.get(id(parent1), -1)
        
        if idx0 >= 0 and idx1 >= 0:
            if idx0 == idx1:  # x*x case
                # ∂²(x²)/∂x² = 2
                d2[(idx0, idx0)] = 2.0
            else:  # x*y case
                # ∂²(xy)/∂x∂y = 1
                if idx0 < idx1:
                    d2[(idx0, idx1)] = 1.0
                else:
                    d2[(idx1, idx0)] = 1.0
    
    elif node.op_tag == 'add' or node.op_tag == 'sub':
        # Addition/Subtraction: all second derivatives are 0
        pass
    
    elif node.op_tag == 'div' and len(node.parents) == 2:
        # Division: f = x / y
        parent0 = node.parents[0][0]  # numerator
        parent1 = node.parents[1][0]  # denominator
        idx0 = var_to_idx.get(id(parent0), -1)
        idx1 = var_to_idx.get(id(parent1), -1)
        
        if idx0 >= 0 and idx1 >= 0:
            y_val = getattr(parent1, 'val', 1.0)
            if y_val != 0:
                if idx0 == idx1:
                    # ∂²(x/x)/∂x² = 0 (when x appears in both num and denom)
                    d2[(idx0, idx0)] = 0.0
                else:
                    # ∂²(x/y)/∂x∂y = -1/y²
                    if idx0 < idx1:
                        d2[(idx0, idx1)] = -1.0 / (y_val * y_val)
                    else:
                        d2[(idx1, idx0)] = -1.0 / (y_val * y_val)
                
                # ∂²(x/y)/∂y² = 2x/y³
                x_val = getattr(parent0, 'val', 1.0)
                d2[(idx1, idx1)] = 2.0 * x_val / (y_val * y_val * y_val)
    
    elif node.op_tag == 'pow' and len(node.parents) == 2:
        # Power: f = x^n
        parent0 = node.parents[0][0]  # base
        parent1 = node.parents[1][0]  # exponent
        idx0 = var_to_idx.get(id(parent0), -1)
        idx1 = var_to_idx.get(id(parent1), -1)

        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 1.0)
            n_val = getattr(parent1, 'val', 2.0)

            # ∂²(x^n)/∂x² = n(n-1)x^(n-2)
            if x_val != 0:
                d2[(idx0, idx0)] = n_val * (n_val - 1) * (x_val ** (n_val - 2))

    elif node.op_tag == 'exp' and len(node.parents) == 1:
        # Exponential: f = exp(x)
        # ∂²exp(x)/∂x² = exp(x)
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)

        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 0.0)
            d2[(idx0, idx0)] = np.exp(x_val)

    elif node.op_tag == 'log' and len(node.parents) == 1:
        # Logarithm: f = log(x)
        # ∂²log(x)/∂x² = -1/x²
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)

        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 1.0)
            if x_val > 0:
                d2[(idx0, idx0)] = -1.0 / (x_val * x_val)

    elif node.op_tag == 'erf' and len(node.parents) == 1:
        # Error function: f = erf(x)
        # ∂²erf(x)/∂x² = -(4x/√π) * exp(-x²)
        parent0 = node.parents[0][0]
        idx0 = var_to_idx.get(id(parent0), -1)

        if idx0 >= 0:
            x_val = getattr(parent0, 'val', 0.0)
            d2[(idx0, idx0)] = -(4.0 * x_val / np.sqrt(np.pi)) * np.exp(-x_val * x_val)

    return d2


def _block1_update(W: SymmSparseAdjMatrix, i: int, preds: List[int],
                   d1: Dict[int, float], n_nodes: int) -> None:
    """
    Block 1: W ← (Φ'ᵢ)ᵀ W Φ'ᵢ

    This implements the complete Block 1 update with two parts:
    1. Standard propagation for pairs within preds
    2. Semi-cross propagation for W(i,r) where r may not be in preds

    The semi-cross propagation is essential for correctly handling cases where
    intermediate variables appear in W matrix entries that need to be propagated
    back to input variables.

    Args:
        W: Working matrix storing accumulated second derivatives
        i: Current node index
        preds: List of predecessor indices (deduplicated)
        d1: First-order derivatives ∂φᵢ/∂xⱼ (already summed for repeated vars)
        n_nodes: Total number of nodes
    """
    # Part 1: Standard three terms for preds×preds
    # Term 1: W(i,i) contribution
    wii = W.get(i, i)
    if wii != 0.0:
        for a, j in enumerate(preds):
            dj = d1.get(j, 0.0)
            if dj == 0.0:
                continue
            for b in range(a, len(preds)):
                k = preds[b]
                dk = d1.get(k, 0.0)
                if dk != 0.0:
                    W.add(j, k, dj * dk * wii)

    # Term 2 & 3: W(i,j) and W(i,k) contributions
    for j_idx, j in enumerate(preds):
        for k_idx in range(j_idx, len(preds)):
            k = preds[k_idx]

            dj = d1.get(j, 0.0)
            dk = d1.get(k, 0.0)

            if dj != 0.0:
                wik = W.get(i, k)
                if wik != 0.0:
                    W.add(j, k, dj * wik)

            if dk != 0.0 and j != k:  # Avoid double-counting diagonal
                wij = W.get(i, j)
                if wij != 0.0:
                    W.add(j, k, dk * wij)

    # Part 2: Semi-cross propagation - propagate ALL W(i,r) to (preds,r)
    # This handles cases where W(i,r) exists but r is NOT in preds
    # Special case: when r IS in preds, only handle diagonal (j==r) because
    # Term 3 skips diagonal due to the "j != k" check
    for r in range(n_nodes):
        if r == i:
            continue

        wir = W.get(i, r)
        if wir == 0.0:
            continue

        # For each predecessor j, propagate W(i,r) to W(j,r)
        for j in preds:
            dj = d1.get(j, 0.0)
            if dj == 0.0:
                continue

            # If r is in preds, only process diagonal (j==r)
            # All off-diagonal pairs are already handled in Part 1
            if r in preds:
                if j != r:
                    continue  # Skip off-diagonal, already handled

            W.add(j, r, dj * wir)

    # Clear row/column i to avoid reuse in subsequent iterations
    W.clear_row_col(i)


def _block2_update(W: SymmSparseAdjMatrix, preds: List[int],
                   d2: Dict[Tuple[int, int], float], vbar: float) -> None:
    """
    Block 2: W += v̄ᵢ * Φ''ᵢ
    
    Adds contribution from local second derivatives.
    
    Args:
        W: Working matrix
        preds: List of predecessor indices
        d2: Second-order derivatives ∂²φᵢ/∂xⱼ∂xₖ
        vbar: Adjoint value at current node
    """
    if vbar == 0.0:
        return  # No contribution if adjoint is zero
    for (j, k), val in d2.items():
        if k<j:
            continue  # only fill upper triangle
        if val != 0.0:
            W.add(j, k, vbar * val)


def _adjoint_update(vbar: List[float], i: int, preds: List[int],
                    d1: Dict[int, float]) -> None:
    """
    Adjoint update: v̄ᵀ ← v̄ᵀ Φ'ᵢ
    
    Standard reverse-mode adjoint propagation.
    
    Args:
        vbar: List of adjoint values
        i: Current node index
        preds: List of predecessor indices
        d1: First-order derivatives
    """
    if vbar[i] == 0.0:
        return  # No update if adjoint is zero
    for j in preds:
        vbar[j] += vbar[i] * d1.get(j, 0.0)


def _extract_input_hessian(W: SymmSparseAdjMatrix, input_indices: List[int]) -> np.ndarray:
    """
    Extract submatrix of W corresponding to input variables.
    
    Args:
        W: Full symmetric sparse matrix
        input_indices: Indices of input variables
        
    Returns:
        Dense Hessian matrix for inputs only
    """
    n = len(input_indices)
    H = np.zeros((n, n))

    for i, idx_i in enumerate(input_indices):
        for j, idx_j in enumerate(input_indices):
            if j < i:
                continue  # only fill upper triangle
            H[i, j] = W.get(idx_i, idx_j)
            if i != j:
                H[j, i] = H[i, j]  # Symmetric
    return H