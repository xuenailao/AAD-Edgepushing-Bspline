"""
EP (with sparse algo4) vs Bumping2 comparison.
"""

import argparse
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
# Add edge_pushing directory for Cython modules
sys.path.insert(0, str(Path(__file__).parent / "aad_edge_pushing" / "edge_pushing"))

from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.pde.models.bspline_model_2d import BSplineModel2D
from aad_edge_pushing.pde.pde_aad_bspline_2d import BS_PDE_AAD_BSpline2D
from aad_edge_pushing.pde.methods.bumping2_bspline_2d import Bumping2BSpline2D
from aad_edge_pushing.edge_pushing.algo4_sparse import algo4_sparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='EP vs Bumping2 comparison test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--grid-m', type=int, default=20,
                       help='M grid size (spatial points)')
    parser.add_argument('--grid-n', type=int, default=10,
                       help='N grid size (time steps)')
    parser.add_argument('--configs', type=str, default='4x4,6x6',
                       help='Comma-separated B-spline configs (e.g., "4x4,6x6,10x10")')
    parser.add_argument('--cython', action='store_true',
                       help='Use Cython-optimized algo4_sparse_openmp (3x faster)')
    return parser.parse_args()


def parse_configs(config_str):
    """Parse '4x4,6x6' into [(4,4), (6,6)]."""
    configs = []
    for cfg in config_str.split(','):
        cfg = cfg.strip()
        if 'x' in cfg:
            n_S, n_T = map(int, cfg.split('x'))
            configs.append((n_S, n_T))
    return configs


def create_model(n_S, n_T):
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


def _parse_coeff_name(name):
    """Parse coefficient name 'w0,3' -> (0, 3)."""
    parts = name[1:].split(',')
    return int(parts[0]), int(parts[1])


def test_ep_sparse(model, coefficients, M, N, use_cython=False):
    """Test EP with sparse algo4."""
    solver = BS_PDE_AAD_BSpline2D(
        bspline_model_2d=model,
        S0=100.0, K=100.0, T=1.0, r=0.05,
        M=M, N_base=N
    )

    # Build tape
    print("  Building tape...", end=" ", flush=True)
    t0 = time.perf_counter()
    global_tape.reset()

    result = solver.solve_pde_with_aad(
        S0_val=100.0,
        coeff_matrix_vals=coefficients,
        compute_hessian=False,
        verbose=False
    )
    t_tape = time.perf_counter() - t0
    print(f"{t_tape:.2f}s")

    # Get inputs (only w* coefficients)
    nodes = global_tape.nodes
    inputs_raw = []
    for node in nodes:
        for parent, _ in node.parents:
            name = getattr(parent, 'name', None)
            if name and name.startswith('w') and parent not in inputs_raw:
                inputs_raw.append(parent)

    # Sort inputs to row-major order to match Bumping2
    inputs_with_idx = [(inp, _parse_coeff_name(inp.name)) for inp in inputs_raw]
    inputs_sorted = sorted(inputs_with_idx, key=lambda x: (x[1][0], x[1][1]))
    inputs = [inp for inp, _ in inputs_sorted]

    output = nodes[-1].out

    # Run sparse algo4 (Python or Cython)
    if use_cython:
        try:
            import algo4_sparse_openmp
            print("  Running Cython algo4...", end=" ", flush=True)
            t0 = time.perf_counter()
            H = algo4_sparse_openmp.algo4_sparse_openmp(output, inputs, n_threads=1, sort_inputs=False)
            t_algo4 = time.perf_counter() - t0
            print(f"{t_algo4:.2f}s")
        except ImportError as e:
            print(f"\n  WARNING: Cython module not found ({e})")
            print("  To enable Cython optimization, run:")
            print("    bash compile_cython.sh")
            print("  Falling back to Python version...")
            print("  Running Python sparse algo4...", end=" ", flush=True)
            t0 = time.perf_counter()
            H = algo4_sparse(output, inputs)
            t_algo4 = time.perf_counter() - t0
            print(f"{t_algo4:.2f}s")
    else:
        print("  Running Python sparse algo4...", end=" ", flush=True)
        t0 = time.perf_counter()
        H = algo4_sparse(output, inputs)
        t_algo4 = time.perf_counter() - t0
        print(f"{t_algo4:.2f}s")

    t_total = t_tape + t_algo4
    return t_total, t_tape, t_algo4, H, result['price']


def test_bumping2(model, coefficients, M, N):
    """Test Bumping2 (diagonal only)."""
    solver = Bumping2BSpline2D(
        bspline_model_2d=model,
        S0=100.0, K=100.0, T=1.0, r=0.05,
        M=M, N=N,
        eps_coeff=0.001
    )

    print("  Running Bumping2...", end=" ", flush=True)
    t0 = time.perf_counter()
    result = solver.compute_diagonal_hessian(coefficients, verbose=False)
    t_bump = time.perf_counter() - t0
    print(f"{t_bump:.2f}s")

    return t_bump, result['hessian_diagonal'], result['price']


def main():
    """Main comparison."""
    args = parse_args()
    M, N = args.grid_m, args.grid_n
    configs = parse_configs(args.configs)

    print("="*70)
    ep_label = "EP (Sparse Cython)" if args.cython else "EP (Sparse Python)"
    print(f"{ep_label} vs Bumping2 Comparison")
    print(f"Grid: M={M}, N={N}")
    print(f"Configs: {args.configs}")
    if args.cython:
        print("Cython optimization: ENABLED (3x faster)")
    print("="*70)

    results = []

    for n_S, n_T in configs:
        n_params = n_S * n_T
        print(f"\n{'='*70}")
        print(f"Testing {n_S}x{n_T} = {n_params} parameters")
        print("="*70)

        model, coefficients = create_model(n_S, n_T)

        # EP with sparse algo4
        print(f"\n[{ep_label}]")
        t_ep, t_tape, t_algo4, H_ep, price_ep = test_ep_sparse(model, coefficients, M, N, use_cython=args.cython)

        # Bumping2
        print("\n[Bumping2]")
        t_bump, H_diag_bump, price_bump = test_bumping2(model, coefficients, M, N)

        # Estimate full Bumping2 time
        n_pde_diag = 2 * n_params + 1
        t_per_pde = t_bump / n_pde_diag
        n_pairs = n_params * (n_params + 1) // 2
        t_bump_full_est = t_per_pde * (4 * n_pairs + 1)

        # Compare diagonal Hessian
        H_diag_ep = np.diag(H_ep)
        diag_match = np.allclose(H_diag_ep, H_diag_bump, rtol=0.1)

        speedup_diag = t_bump / t_ep
        speedup_full = t_bump_full_est / t_ep

        print(f"\n[Results]")
        print(f"  EP total: {t_ep:.2f}s (tape: {t_tape:.2f}s, algo4: {t_algo4:.2f}s)")
        print(f"  Bumping2 diag: {t_bump:.2f}s")
        print(f"  Bumping2 full (est): {t_bump_full_est:.2f}s")
        print(f"  Speedup vs diag: {speedup_diag:.2f}x")
        print(f"  Speedup vs full: {speedup_full:.2f}x")
        print(f"  Diagonal Hessian match: {'Yes' if diag_match else 'No'}")
        print(f"  Price EP: {price_ep:.6f}")
        print(f"  Price Bump: {price_bump:.6f}")

        results.append({
            'n_params': n_params,
            't_ep': t_ep,
            't_bump_diag': t_bump,
            't_bump_full_est': t_bump_full_est,
            'speedup_diag': speedup_diag,
            'speedup_full': speedup_full
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'n_params':>8} | {'EP (s)':>10} | {'Bump diag':>10} | {'Bump full':>12} | {'Speedup':>10}")
    print("-"*70)
    for r in results:
        s = "FASTER" if r['speedup_diag'] > 1 else "slower"
        print(f"{r['n_params']:>8} | {r['t_ep']:>10.2f} | {r['t_bump_diag']:>10.2f} | {r['t_bump_full_est']:>12.2f} | {r['speedup_diag']:>8.2f}x ({s})")


if __name__ == "__main__":
    main()
