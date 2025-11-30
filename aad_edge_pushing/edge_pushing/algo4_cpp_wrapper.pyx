# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
Cython wrapper for pure C++ OpenMP parallelized Algorithm 4.

This wrapper:
1. Converts Python tape to C++ structures
2. Calls pure C++ implementation
3. Converts C++ Hessian back to NumPy array

All computation happens in C++ with OpenMP, completely avoiding GIL.
"""

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.string cimport string
from typing import List

# External C++ declarations
cdef extern from "algo4_cpp_parallel.hpp" namespace "algo4":
    cdef cppclass Node:
        int id
        string op_tag
        vector[int] parent_ids
        vector[double] parent_derivs
        vector[double] parent_vals
        double out_val

    cdef cppclass Tape:
        vector[Node] nodes
        unordered_map[int, int] var_to_idx

    vector[vector[double]] algo4_cpp_parallel(
        const Tape& tape,
        const vector[int]& input_indices,
        int output_idx,
        int n_threads
    ) nogil


def algo4_cpp_openmp(output, inputs: List, int n_threads=16):
    """
    Compute Hessian using pure C++ OpenMP-parallelized Algorithm 4.

    This completely avoids Python's GIL by:
    - Converting tape to C++ structures
    - Running all computation in C++
    - Only using Python for I/O

    Args:
        output: Output ADVar (scalar function output)
        inputs: List of input ADVars
        n_threads: Number of OpenMP threads (default: 16)

    Returns:
        Dense Hessian matrix of shape (n_inputs, n_inputs)
    """
    from ..aad.core.tape import global_tape

    # Create index mapping
    cdef dict var_to_idx = {}
    cdef int idx = 0

    # Map inputs first
    cdef list input_indices_list = []
    for inp in inputs:
        var_to_idx[id(inp)] = idx
        input_indices_list.append(idx)
        idx += 1

    # Map all nodes
    for node in global_tape.nodes:
        if id(node.out) not in var_to_idx:
            var_to_idx[id(node.out)] = idx
            idx += 1
        for parent, _ in node.parents:
            if id(parent) not in var_to_idx:
                var_to_idx[id(parent)] = idx
                idx += 1

    # Make sure output is mapped
    if id(output) not in var_to_idx:
        var_to_idx[id(output)] = idx
        idx += 1

    cdef int output_idx = var_to_idx[id(output)]

    # Convert to C++ structures
    cdef Tape cpp_tape = _convert_tape_to_cpp(global_tape, var_to_idx)
    cdef vector[int] cpp_input_indices = input_indices_list
    cdef vector[vector[double]] cpp_hessian

    # Call C++ implementation (releases GIL!)
    with nogil:
        cpp_hessian = algo4_cpp_parallel(cpp_tape, cpp_input_indices, output_idx, n_threads)

    # Convert result to NumPy
    return _cpp_to_numpy(cpp_hessian)


cdef Tape _convert_tape_to_cpp(tape, dict var_to_idx):
    """Convert Python tape to C++ Tape structure."""
    cdef Tape cpp_tape
    cdef int parent_idx, node_id, node_idx
    cdef double parent_val
    cdef Node temp_node

    # Pre-allocate space to avoid push_back issues
    cdef int n_nodes = len(tape.nodes)
    cpp_tape.nodes.reserve(n_nodes)

    node_idx = 0
    for py_node in tape.nodes:
        node_id = var_to_idx[id(py_node.out)]

        # Create temporary node
        temp_node.id = node_id
        temp_node.op_tag = (py_node.op_tag.encode('utf-8')
                           if hasattr(py_node, 'op_tag') else b'')
        temp_node.out_val = getattr(py_node.out, 'val', 0.0)

        # Clear vectors for reuse
        temp_node.parent_ids.clear()
        temp_node.parent_derivs.clear()
        temp_node.parent_vals.clear()

        # Add parents
        for parent, deriv in py_node.parents:
            parent_idx = var_to_idx[id(parent)]
            parent_val = getattr(parent, 'val', 0.0)
            temp_node.parent_ids.push_back(parent_idx)
            temp_node.parent_derivs.push_back(float(deriv))
            temp_node.parent_vals.push_back(parent_val)

        # Add node to tape (using copy)
        cpp_tape.nodes.push_back(temp_node)
        cpp_tape.var_to_idx[node_id] = node_idx
        node_idx += 1

    return cpp_tape


cdef np.ndarray _cpp_to_numpy(const vector[vector[double]]& cpp_matrix):
    """Convert C++ 2D vector to NumPy array."""
    cdef int n = cpp_matrix.size()
    cdef int m = cpp_matrix[0].size() if n > 0 else 0
    cdef np.ndarray[np.float64_t, ndim=2] result = np.zeros((n, m), dtype=np.float64)

    cdef int i, j
    for i in range(n):
        for j in range(m):
            result[i, j] = cpp_matrix[i][j]

    return result
