# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
OpenMP parallelized version V3 - 真正绕开 GIL！

关键改进（相比 v2）:
1. 使用 C++ vector 代替 Python list 存储更新
2. 调用 cdef nogil 方法代替 cpdef 方法
3. 整个并行循环在 nogil 区域内
4. 使用 OpenMP critical section 而不是 GIL

预期: 4-8× 加速（相比单线程）

技术创新:
- add_nogil() 和 get_nogil() 方法 (真正 nogil)
- C++ vector<tuple<int, int, double>> 存储更新
- OpenMP critical section 保护共享数据
- 真正的线程并行（无 GIL 竞争）

作者: Claude Code
日期: 2025-11-12
"""

import numpy as np
cimport numpy as np
from typing import List
from libc.math cimport exp, sqrt
from cython.parallel cimport prange, parallel
cimport openmp
from libcpp.vector cimport vector
from libcpp.pair cimport pair

# Import C++ optimized sparse matrix with nogil support
from symm_sparse_adjlist_cpp cimport SymmSparseAdjListCpp

# Define C++ types for updates
ctypedef pair[int, pair[int, double]] UpdateType  # (j, (k, val))


def algo4_openmp_v3(output, inputs: List, int n_threads=16):
    """
    Compute Hessian using真正的 nogil OpenMP parallelization.

    Key improvements over v2:
    - C++ vector for storing updates (no Python list)
    - cdef nogil methods (no GIL needed)
    - OpenMP critical section (not GIL lock)
    - True thread parallelism

    Args:
        output: Output ADVar (scalar function output)
        inputs: List of input ADVars
        n_threads: Number of OpenMP threads (default: 16)

    Returns:
        Dense Hessian matrix of shape (n_inputs, n_inputs)
    """
    from aad_edge_pushing.aad.core.tape import global_tape

    # Set OpenMP thread count
    openmp.omp_set_num_threads(n_threads)

    mapping = _create_index_mapping(global_tape, inputs, output)
    var_to_idx = mapping['var_to_idx']
    idx_to_var = mapping['idx_to_var']
    input_indices = mapping['input_indices']
    output_idx = mapping['output_idx']
    n_nodes = mapping['n_nodes']

    # Initialize W matrix and adjoint vector
    W = SymmSparseAdjListCpp(n_nodes)
    vbar = [0.0] * n_nodes
    vbar[output_idx] = 1.0  # Seed adjoint for output

    # Reverse sweep with真正 nogil
    _reverse_sweep_v3(global_tape.nodes, W, vbar, var_to_idx, n_threads)

    return _extract_input_hessian(W, input_indices)


cdef void _reverse_sweep_v3(list nodes, SymmSparseAdjListCpp W, list vbar, dict var_to_idx, int n_threads):
    """
    V3 反向传播 - 真正 nogil 并行化.

    关键改进:
    1. 收集更新到 C++ vector (不是 Python list)
    2. 调用 nogil 方法
    3. 整个应用更新过程无 GIL
    """
    cdef int node_idx, n_nodes
    n_nodes = len(nodes)

    # Main reverse sweep - 顺序遍历节点（依赖关系）
    for node_idx in range(n_nodes - 1, -1, -1):
        node = nodes[node_idx]
        i = var_to_idx[id(node.out)]

        # Get predecessors and derivatives (Python 操作，需要 GIL)
        preds = _get_predecessors(node, var_to_idx)
        preds = sorted(set(preds))
        d1 = _get_first_derivatives(node, var_to_idx)
        d2 = _compute_second_derivatives(node, var_to_idx)

        # PUSHING STAGE - 真正 nogil 并行化！
        _pushing_stage_v3(W, i, preds, d1, n_threads)

        # CREATING STAGE
        vbar_i = vbar[i]
        if vbar_i != 0:
            _creating_stage_v3(W, preds, d2, vbar_i, n_threads)

            # ADJOINT STAGE (顺序，快速)
            for j in preds:
                dj = d1.get(j, 0.0)
                if dj != 0.0:
                    vbar[j] += vbar_i * dj


cdef void _pushing_stage_v3(SymmSparseAdjListCpp W, int i, list preds, dict d1, int n_threads):
    """
    V3 pushing stage - 收集更新到 C++ vector.
    """
    cdef int p, j, k, a, b
    cdef double w_pi, dj, dk, update_val
    cdef int n_preds = len(preds)

    # Get all neighbors (Python 操作)
    neighbors = W.get_neighbors(i)

    if len(neighbors) == 0:
        return

    # 收集所有更新到 C++ vector
    cdef vector[UpdateType] updates

    for p, w_pi in neighbors:
        if p == i:
            # Diagonal case
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
                        updates.push_back(UpdateType(j, pair[int, double](k, update_val)))
        else:
            # Off-diagonal case
            for a in range(n_preds):
                j = preds[a]
                dj = d1.get(j, 0.0)
                if dj == 0.0:
                    continue

                if j == p:
                    update_val = 2.0 * dj * w_pi
                    updates.push_back(UpdateType(p, pair[int, double](p, update_val)))
                else:
                    update_val = dj * w_pi
                    updates.push_back(UpdateType(p, pair[int, double](j, update_val)))

    # 并行应用更新 - 真正 nogil！
    cdef int n_updates = updates.size()

    if n_updates > 20:  # 只有足够多更新时才并行
        _apply_updates_nogil(W, updates, n_threads)
    else:
        # 更新太少，直接顺序执行（使用 Python 方法）
        for idx in range(n_updates):
            j = updates[idx].first
            k = updates[idx].second.first
            val = updates[idx].second.second
            W.add(j, k, val)

    # Clear row/column i
    W.clear_row_col(i)


cdef void _apply_updates_nogil(
    SymmSparseAdjListCpp W,
    vector[UpdateType] updates,
    int n_threads
) nogil:
    """
    真正的 nogil 并行更新！

    关键改进:
    - 整个函数声明为 nogil
    - 使用 C++ vector (不是 Python list)
    - 调用 cdef nogil 方法
    - OpenMP critical section (不是 GIL)
    """
    cdef int idx, j, k
    cdef double val
    cdef int n_updates = updates.size()

    # 真正的并行 - 无 GIL！
    for idx in prange(n_updates, nogil=True, num_threads=n_threads, schedule='dynamic'):
        j = updates[idx].first
        k = updates[idx].second.first
        val = updates[idx].second.second

        # 调用 nogil 方法 - 仍需要串行化（critical）
        # 但无 GIL，只有 C++ 级别的锁
        W.add_nogil(j, k, val)


cdef void _creating_stage_v3(SymmSparseAdjListCpp W, list preds, dict d2, double vbar, int n_threads):
    """
    V3 creating stage - 真正 nogil 并行化.
    """
    if vbar == 0.0:
        return

    # 收集更新到 C++ vector
    cdef vector[UpdateType] updates

    for (j, k), val in d2.items():
        if k >= j and val != 0.0:
            update_val = vbar * val
            updates.push_back(UpdateType(j, pair[int, double](k, update_val)))

    cdef int n_updates = updates.size()

    if n_updates > 20:  # 只有足够多更新时才并行
        _apply_updates_nogil(W, updates, n_threads)
    else:
        for idx in range(n_updates):
            j = updates[idx].first
            k = updates[idx].second.first
            val = updates[idx].second.second
            W.add(j, k, val)


# ============================================================================
# Helper functions (same as previous versions)
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
