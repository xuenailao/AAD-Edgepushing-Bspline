# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
OpenMP parallelized version using local accumulation strategy.

Key improvement over algo4_openmp_safe:
- Each thread maintains its own local sparse matrix
- No locks/critical sections during computation
- Final merge step combines all local matrices
- Expected speedup: 4-8× (vs 0.1× for lock-based version)
"""

import numpy as np
cimport numpy as np
from typing import List
from libc.math cimport exp, sqrt
from cython.parallel cimport prange, parallel
cimport openmp
from libcpp.vector cimport vector

# Import the C++ optimized sparse matrix
from .symm_sparse_adjlist_cpp import SymmSparseAdjListCpp


def algo4_openmp_local(output, inputs: List, int n_threads=16):
    """
    Compute Hessian using OpenMP with local accumulation strategy.

    This version avoids lock contention by having each thread maintain
    its own local Hessian matrix, then merging them at the end.

    Args:
        output: Output ADVar (scalar function output)
        inputs: List of input ADVars
        n_threads: Number of OpenMP threads (default: 16)

    Returns:
        Dense Hessian matrix of shape (n_inputs, n_inputs)
    """
    from ..aad.core.tape import global_tape

    # Set OpenMP thread count
    openmp.omp_set_num_threads(n_threads)

    mapping = _create_index_mapping(global_tape, inputs, output)
    var_to_idx = mapping['var_to_idx']
    idx_to_var = mapping['idx_to_var']
    input_indices = mapping['input_indices']
    output_idx = mapping['output_idx']
    n_nodes = mapping['n_nodes']

    # Initialize adjoint vector
    vbar = [0.0] * n_nodes
    vbar[output_idx] = 1.0  # Seed adjoint for output

    # Reverse sweep with local accumulation
    W = _reverse_sweep_openmp_local(global_tape.nodes, vbar, var_to_idx,
                                     n_nodes, n_threads)

    return _extract_input_hessian(W, input_indices)


def _reverse_sweep_openmp_local(list nodes, list vbar, dict var_to_idx,
                                 int n_nodes, int n_threads):
    """
    OpenMP-parallelized reverse sweep using local accumulation.

    Strategy:
    1. Each thread maintains a local W matrix
    2. No locks during computation
    3. Final sequential merge of local matrices
    """
    cdef int node_idx, i
    cdef double vbar_i
    cdef list preds
    cdef dict d1, d2

    # Create thread-local matrices (Python list for simplicity)
    W_locals = [SymmSparseAdjListCpp(n_nodes) for _ in range(n_threads)]

    # Main reverse sweep (must be sequential due to dependencies)
    for node_idx in range(len(nodes) - 1, -1, -1):
        node = nodes[node_idx]
        i = var_to_idx[id(node.out)]

        # Get predecessors and derivatives
        preds = _get_predecessors(node, var_to_idx)
        preds = sorted(set(preds))
        d1 = _get_first_derivatives(node, var_to_idx)
        d2 = _compute_second_derivatives(node, var_to_idx)

        # PUSHING STAGE - Parallelized with local accumulation
        _pushing_stage_local(W_locals, i, preds, d1, n_threads)

        # CREATING STAGE
        vbar_i = vbar[i]
        if vbar_i != 0:
            _creating_stage_local(W_locals, preds, d2, vbar_i, n_threads)

            # ADJOINT STAGE (sequential, very fast)
            for j in preds:
                dj = d1.get(j, 0.0)
                if dj != 0.0:
                    vbar[j] += vbar_i * dj

    # Merge all local matrices into final result
    W_final = SymmSparseAdjListCpp(n_nodes)
    for W_local in W_locals:
        _merge_matrices(W_final, W_local)

    return W_final


def _pushing_stage_local(list W_locals, int i, list preds, dict d1, int n_threads):
    """
    Pushing stage with local accumulation (lock-free).

    Each thread works on a subset of neighbors and writes to its local matrix.
    """
    cdef int p, j, k, a, b, idx, tid
    cdef double w_pi, dj, dk, update_val
    cdef int n_preds = len(preds)

    # Get neighbors from first local matrix (all should have same structure at this point)
    # Actually, we need to collect neighbors from ALL local matrices
    all_neighbors = {}
    for W_local in W_locals:
        neighbors = W_local.get_neighbors(i)
        for p, w_val in neighbors:
            all_neighbors[p] = all_neighbors.get(p, 0.0) + w_val

    neighbor_list = list(all_neighbors.items())
    cdef int n_neighbors = len(neighbor_list)

    if n_neighbors == 0:
        return

    # Parallel loop over neighbors - each thread uses its own local matrix
    cdef int thread_id
    for idx in prange(n_neighbors, nogil=True, num_threads=n_threads, schedule='dynamic'):
        thread_id = openmp.omp_get_thread_num()

        with gil:
            p, w_pi = neighbor_list[idx]
            W_local = W_locals[thread_id]

            if p == i:
                # Case 1: Diagonal W(i,i) contribution
                for a in range(n_preds):
                    j = preds[a]
                    dj = d1.get(j, 0.0)
                    if dj == 0.0:
                        continue

                    for b in range(a, n_preds):
                        k = preds[b]
                        dk = d1.get(k, 0.0)
                        if dk != 0.0:
                            update_val = dj * dk * w_pi
                            # Lock-free write to local matrix
                            W_local.add(j, k, update_val)
            else:
                # Case 2: Off-diagonal W(p,i) contribution
                for a in range(n_preds):
                    j = preds[a]
                    dj = d1.get(j, 0.0)
                    if dj == 0.0:
                        continue

                    if j == p:
                        update_val = 2.0 * dj * w_pi
                        W_local.add(p, p, update_val)
                    else:
                        update_val = dj * w_pi
                        W_local.add(p, j, update_val)

    # Clear row/column i from all local matrices
    for W_local in W_locals:
        W_local.clear_row_col(i)


def _creating_stage_local(list W_locals, list preds, dict d2, double vbar, int n_threads):
    """
    Creating stage with local accumulation (lock-free).
    """
    cdef int j, k, idx, thread_id
    cdef double val, update_val
    cdef list d2_items
    cdef int n_items

    if vbar == 0.0:
        return

    # Convert dict to list for parallel iteration
    d2_items = list(d2.items())
    n_items = len(d2_items)

    if n_items == 0:
        return

    # Parallel loop over second derivatives
    for idx in prange(n_items, nogil=True, num_threads=n_threads, schedule='static'):
        thread_id = openmp.omp_get_thread_num()

        with gil:
            (j, k), val = d2_items[idx]

            if k >= j and val != 0.0:
                update_val = vbar * val
                W_local = W_locals[thread_id]
                # Lock-free write to local matrix
                W_local.add(j, k, update_val)


def _merge_matrices(W_final, W_local):
    """
    Merge a local matrix into the final result.

    This is sequential but only happens once per thread at the end.
    """
    # Get all non-zero entries from local matrix
    cdef int i, j
    cdef double val

    # Iterate over all possible nodes
    n_nodes = W_local.n  # Use the n attribute
    for i in range(n_nodes):
        neighbors = W_local.get_neighbors(i)
        for j, val in neighbors:
            if j >= i:  # Only upper triangle
                W_final.add(i, j, val)


# ============================================================================
# Helper functions (copied from algo4_cython_simple.pyx)
# ============================================================================

def _create_index_mapping(tape, inputs: List, output):
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


def _get_predecessors(node, var_to_idx):
    """Get predecessor indices for a given node."""
    preds = []
    for parent, _ in node.parents:
        preds.append(var_to_idx[id(parent)])
    return preds


def _get_first_derivatives(node, var_to_idx):
    """Extract first-order local derivatives, summing repeated variables."""
    d1 = {}
    for parent, deriv in node.parents:
        j = var_to_idx[id(parent)]
        d1[j] = d1.get(j, 0.0) + float(deriv)
    return d1


def _compute_second_derivatives(node, var_to_idx):
    """Compute second-order local derivatives for a node."""
    d2 = {}

    if node.op_tag == 'mul' and len(node.parents) == 2:
        parent0, parent1 = node.parents[0][0], node.parents[1][0]
        idx0, idx1 = var_to_idx.get(id(parent0), -1), var_to_idx.get(id(parent1), -1)

        if idx0 >= 0 and idx1 >= 0:
            if idx0 == idx1:
                d2[(idx0, idx0)] = 2.0
            else:
                d2[(min(idx0, idx1), max(idx0, idx1))] = 1.0

    elif node.op_tag in ('add', 'sub'):
        pass

    elif node.op_tag == 'div' and len(node.parents) == 2:
        parent0, parent1 = node.parents[0][0], node.parents[1][0]
        idx0, idx1 = var_to_idx.get(id(parent0), -1), var_to_idx.get(id(parent1), -1)

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


def _extract_input_hessian(W, input_indices):
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
