"""
Test OpenMP parallelization speedup for sparse algo4.

Compares:
1. algo4_sparse (pure Python sparse, single thread)
2. algo4_sparse_openmp (Cython + OpenMP, multiple threads)
"""

import sys
from pathlib import Path
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
# Add edge_pushing directory for compiled modules
sys.path.insert(0, str(Path(__file__).parent / "aad_edge_pushing" / "edge_pushing"))

from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.pde.models.bspline_model_2d import BSplineModel2D
from aad_edge_pushing.pde.pde_aad_bspline_2d import BS_PDE_AAD_BSpline2D


def create_model(n_S=6, n_T=6):
    """Create B-spline model."""
    degree_S, degree_T = 3, 2
    k_S = n_S - degree_S + 1
    k_T = n_T - degree_T + 1

    knots_S = np.linspace(50.0, 150.0, k_S)
    knots_T = np.linspace(0.0, 1.0, k_T)
    coefficients = np.full((n_S, n_T), 0.2)

    model = BSplineModel2D(
        knots_S=knots_S, knots_T=knots_T,
        coefficients=coefficients,
        degree_S=degree_S, degree_T=degree_T,
        S_min=50.0, S_max=150.0,
        T_min=0.0, T_max=1.0
    )
    return model, coefficients


def build_tape(model, coefficients, M=20, N=10):
    """Build computation tape."""
    solver = BS_PDE_AAD_BSpline2D(
        bspline_model_2d=model,
        S0=100.0, K=100.0, T=1.0, r=0.05,
        M=M, N_base=N
    )

    global_tape.reset()
    result = solver.solve_pde_with_aad(
        S0_val=100.0,
        coeff_matrix_vals=coefficients,
        compute_hessian=False,
        verbose=False
    )

    # Extract inputs (w* coefficients)
    nodes = global_tape.nodes
    inputs_raw = []
    for node in nodes:
        for parent, _ in node.parents:
            name = getattr(parent, 'name', None)
            if name and name.startswith('w') and parent not in inputs_raw:
                inputs_raw.append(parent)

    # Sort to row-major order
    def parse_name(inp):
        name = inp.name
        parts = name[1:].split(',')
        return int(parts[0]), int(parts[1])

    inputs_sorted = sorted(inputs_raw, key=parse_name)
    output = nodes[-1].out

    return output, inputs_sorted, result['price']


def test_sparse_python(output, inputs):
    """Test pure Python sparse algo4."""
    from aad_edge_pushing.edge_pushing.algo4_sparse import algo4_sparse

    print("  Python sparse...", end=" ", flush=True)
    t0 = time.perf_counter()
    H = algo4_sparse(output, inputs, sort_inputs=False)
    t = time.perf_counter() - t0
    print(f"{t:.2f}s")
    return H, t


def test_sparse_openmp(output, inputs, n_threads):
    """Test Cython + OpenMP sparse algo4."""
    # Import directly from compiled module
    import algo4_sparse_openmp

    print(f"  OpenMP ({n_threads} threads)...", end=" ", flush=True)
    t0 = time.perf_counter()
    H = algo4_sparse_openmp.algo4_sparse_openmp(output, inputs, n_threads=n_threads, sort_inputs=False)
    t = time.perf_counter() - t0
    print(f"{t:.2f}s")
    return H, t


def main():
    """Run performance comparison."""
    configs = [(6, 6), (8, 8)]
    thread_counts = [1, 2, 4, 8]

    print("=" * 70)
    print("OpenMP Parallelization Speedup Test")
    print("=" * 70)

    for n_S, n_T in configs:
        n_params = n_S * n_T
        print(f"\n{'='*70}")
        print(f"Configuration: {n_S}x{n_T} = {n_params} parameters")
        print("=" * 70)

        # Build tape once
        print("Building computation tape...")
        model, coefficients = create_model(n_S, n_T)
        output, inputs, price = build_tape(model, coefficients, M=20, N=10)
        print(f"Tape built: {len(inputs)} inputs, {len(global_tape.nodes)} nodes\n")

        # Test Python sparse (baseline)
        H_python, t_python = test_sparse_python(output, inputs)

        # Test OpenMP with different thread counts
        results = []
        for n_threads in thread_counts:
            H_omp, t_omp = test_sparse_openmp(output, inputs, n_threads)

            # Verify correctness
            if not np.allclose(H_omp, H_python, rtol=1e-8, atol=1e-10):
                print(f"    WARNING: Results don't match for {n_threads} threads!")

            speedup = t_python / t_omp
            results.append((n_threads, t_omp, speedup))

        # Summary
        print(f"\n  Summary:")
        print(f"  {'Threads':>8} | {'Time (s)':>10} | {'Speedup':>10}")
        print(f"  {'-'*35}")
        print(f"  {'Python':>8} | {t_python:>10.2f} | {'1.00x':>10}")
        for n_threads, t_omp, speedup in results:
            print(f"  {n_threads:>8} | {t_omp:>10.2f} | {speedup:>9.2f}x")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
