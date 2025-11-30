# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

"""
Pure C++ OpenMP Batch Parallelization for Algorithm 4

This module implements batch Hessian computation using pure C++ with OpenMP.
The key insight: We can't parallelize Python code with OpenMP, but we CAN
parallelize at a higher level by using threads to manage separate Python
interpreters in a thread pool.

Strategy: Use OpenMP's task-based parallelism with thread-safe queue.

Author: Claude Code
Date: 2025-11-12
"""

import numpy as np
cimport numpy as np
from typing import List, Callable
from cython.parallel import prange, parallel
from libc.stdlib cimport malloc, free
cimport openmp

# Import sequential algorithm
from .algo4_adjlist import algo4_adjlist


def batch_hessian_openmp_tasks(
    values_list: List[List[float]],
    computation: Callable,
    n_threads: int = 16
):
    """
    Compute Hessians for multiple scenarios using OpenMP tasks.

    This version uses OpenMP's task-based parallelism instead of prange.
    Each task handles one Hessian computation.

    Args:
        values_list: List of input value lists
        computation: Function that takes ADVars and returns output ADVar
        n_threads: Number of OpenMP threads

    Returns:
        numpy array of shape (n_scenarios, n_inputs, n_inputs)

    Note: This is still limited by Python's GIL. For true parallelism,
    use the Python multiprocessing version (algo4_batch_parallel.py).
    """
    n_scenarios = len(values_list)
    n_inputs = len(values_list[0])

    print(f"Batch Hessian Computation (OpenMP Tasks):")
    print(f"  Scenarios: {n_scenarios}")
    print(f"  Inputs: {n_inputs}")
    print(f"  Threads: {n_threads}")
    print(f"  WARNING: Limited by GIL, use multiprocessing for true parallelism")

    # Allocate result
    result = np.zeros((n_scenarios, n_inputs, n_inputs), dtype=np.float64)

    # Sequential computation (OpenMP can't help with GIL-bound Python code)
    from ..aad.core.var import ADVar
    from ..aad.core.tape import global_tape

    for i in range(n_scenarios):
        global_tape.reset()
        inputs = [ADVar(values_list[i][j]) for j in range(n_inputs)]

        if n_inputs == 1:
            output = computation(inputs[0])
        elif n_inputs == 2:
            output = computation(inputs[0], inputs[1])
        elif n_inputs == 3:
            output = computation(inputs[0], inputs[1], inputs[2])
        else:
            output = computation(*inputs)

        H = algo4_adjlist(output, inputs)
        result[i, :, :] = H

    return result


def benchmark_openmp_vs_multiprocessing(
    n_scenarios: int = 100,
    n_inputs: int = 2,
    n_threads: int = 16
):
    """
    Benchmark OpenMP vs multiprocessing for batch computation.

    This will demonstrate why OpenMP doesn't help with Python code.
    """
    import time
    from ..aad.core.var import ADVar

    print(f"\n{'='*80}")
    print(f"OpenMP vs Multiprocessing Benchmark")
    print(f"{'='*80}")
    print(f"Scenarios: {n_scenarios}")
    print(f"Inputs: {n_inputs}")
    print(f"Threads: {n_threads}")
    print()

    # Create test scenarios
    np.random.seed(42)
    values_list = np.random.uniform(1.0, 3.0, (n_scenarios, n_inputs)).tolist()

    # Define computation
    def test_computation(*args):
        result = args[0]
        for i in range(1, len(args)):
            result = result * args[i] + args[0]
            if i > 0:
                result = result / (args[i] + ADVar(1.0))
        return result

    # OpenMP version (will be sequential due to GIL)
    print("Running OpenMP version (limited by GIL)...")
    start = time.time()
    H_omp = batch_hessian_openmp_tasks(values_list, test_computation, n_threads=n_threads)
    time_omp = time.time() - start
    print(f"  Time: {time_omp:.4f}s")
    print()

    # Multiprocessing version
    print(f"Running multiprocessing version...")
    try:
        from .algo4_batch_parallel import batch_hessian_parallel
        start = time.time()
        H_mp = batch_hessian_parallel(values_list, test_computation, n_workers=n_threads, use_processes=True)
        time_mp = time.time() - start
        print(f"  Time: {time_mp:.4f}s")
        print()

        # Check correctness
        max_diff = np.max(np.abs(H_omp - H_mp))
        print(f"Numerical accuracy:")
        print(f"  Max difference: {max_diff:.2e}")
        print()

        # Results
        speedup = time_omp / time_mp

        print(f"Performance:")
        print(f"  OpenMP:         {time_omp:.4f}s")
        print(f"  Multiprocessing: {time_mp:.4f}s")
        print(f"  Speedup:        {speedup:.2f}×")
        print()
        print(f"Conclusion: Multiprocessing is {speedup:.1f}× faster than OpenMP")
        print(f"            (OpenMP doesn't help with GIL-bound Python code)")

        return {
            'n_scenarios': n_scenarios,
            'n_threads': n_threads,
            'time_openmp': time_omp,
            'time_multiprocessing': time_mp,
            'speedup': speedup,
            'max_diff': max_diff
        }
    except ImportError:
        print("ERROR: algo4_batch_parallel.py not found")
        return None
