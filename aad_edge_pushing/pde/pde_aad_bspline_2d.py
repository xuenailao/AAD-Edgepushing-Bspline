"""
Black-Scholes PDE Solver with 2D B-Spline Volatility Surface and AAD Edge-Pushing

This module implements a PDE solver where the volatility is parameterized as a 2D
B-spline function σ(S,t) = Σᵢ Σⱼ wᵢⱼ·Bᵢ(S)·Bⱼ(t), with all coefficients {wᵢⱼ} as ADVars.

Key Features:
- 2D B-spline volatility: σ(S,t) computed locally at each grid point (S_i, t_n)
- Time-varying volatility: Full term structure modeling
- Sparse Hessian: Edge-pushing exploits compact support in both dimensions
- Natural cubic spline interpolation for final price extraction
- Crank-Nicolson time stepping with ADVar tridiagonal solver

Integration:
This extends BS_PDE_AAD_BSpline to handle 2D tensor product B-splines instead
of 1D spatial-only B-splines.

Computational Complexity:
- Forward pass: O(M × N × (degree_S+1) × (degree_T+1))
- Backward pass (edge-pushing): O(n_params × bandwidth)
- For typical parameters (M=200, N=50, 14×8 params): ~1-2 seconds
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from .pde_config import PDEConfig


class BS_PDE_AAD_BSpline2D:
    """
    Black-Scholes PDE solver with 2D B-spline volatility parameterization.

    The volatility σ(S,t) = Σᵢ Σⱼ wᵢⱼ·Bᵢ(S)·Bⱼ(t) is evaluated locally at each
    grid point (S_i, t_n). All coefficients wᵢⱼ are ADVars, enabling edge-pushing
    to compute the full Hessian matrix ∂²V/∂wᵢⱼ∂wₖₗ in a single backward pass.
    """

    def __init__(self, bspline_model_2d, S0: float, K: float, T: float, r: float,
                 M: int = 200, N_base: int = 50,
                 use_adaptive_Smax: bool = True, sigma_margin: float = 0.1):
        """
        Initialize BS PDE solver with 2D B-spline volatility.

        Args:
            bspline_model_2d: BSplineModel2D instance
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            M: Number of spatial grid points
            N_base: Base number of time steps
            use_adaptive_Smax: Use volatility-adaptive S_max
            sigma_margin: Safety margin for grid sizing
        """
        from aad_edge_pushing.pde.models.bspline_model_2d import BSplineModel2D

        if not isinstance(bspline_model_2d, BSplineModel2D):
            raise TypeError("bspline_model_2d must be a BSplineModel2D instance")

        self.bspline_model_2d = bspline_model_2d
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.M = M
        self.N_base = N_base

        # Estimate typical volatility for grid sizing
        # Use max volatility across time at S0 for conservative grid
        t_samples = np.linspace(0, T, 10)
        S_samples = np.full_like(t_samples, S0)
        sigma_samples = bspline_model_2d.evaluate(S_samples, t_samples)
        sigma_typical = np.max(sigma_samples)

        # Adaptive S_max with margin
        if use_adaptive_Smax:
            S_max = PDEConfig.compute_unified_Smax(K, T, sigma_typical, r, sigma_margin)
        else:
            S_max = 5.0 * K

        self.S_max = S_max

        # Create log-space grid
        S_min = 1e-3
        self.x_grid, self.S_grid, self.dx = PDEConfig.create_log_space_grid(S_min, S_max, M)

        # Time grid
        self.N = N_base
        self.dt = T / self.N
        self.t_grid = np.linspace(0, T, N_base + 1)

        # CFL analysis using typical volatility
        alpha_typical = (sigma_typical**2 / 2.0) / (self.dx**2)
        dt_critical = (self.dx**2) / (2.0 * alpha_typical)
        self.cfl_ratio = self.dt / dt_critical

        # Crank-Nicolson parameter
        self.phi = 0.5

        # Cache for active parameters (computed lazily)
        self._active_params_cache = None

    def get_active_parameters(self) -> set:
        """
        Get the set of active parameter indices (i, j) that contribute to the price.

        Due to B-spline compact support, only parameters whose basis functions
        are non-zero at some point in the PDE grid will affect the price.

        Returns:
            Set of (i, j) tuples for active parameters
        """
        if self._active_params_cache is not None:
            return self._active_params_cache

        n_S, n_T = self.bspline_model_2d.coefficients.shape
        active_params = set()

        # Check all interior grid points at all time steps
        for S in self.S_grid[1:-1]:
            for t in self.t_grid[1:]:
                idx_S, idx_T = self.bspline_model_2d.get_active_basis_indices(S, t)
                for i in idx_S:
                    for j in idx_T:
                        if i < n_S and j < n_T:
                            active_params.add((i, j))

        self._active_params_cache = active_params
        return active_params

    def _terminal_condition(self) -> np.ndarray:
        """Terminal payoff: max(S - K, 0) for European call."""
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_condition_left(self, t: float) -> float:
        """Left boundary: S ≈ 0, call option worthless."""
        return 0.0

    def _boundary_condition_right(self, t: float) -> float:
        """Right boundary: S → ∞, call option ≈ S - K*exp(-r*(T-t))."""
        tau = self.T - t  # Time to maturity
        return self.S_grid[-1] - self.K * np.exp(-self.r * tau)

    def _evaluate_sigma_squared_2d(self, S_points: np.ndarray, t: float,
                                   coeff_matrix_advars: np.ndarray) -> List[ADVar]:
        """
        Evaluate σ²(S,t) at given spatial points for fixed time using 2D B-spline.

        This is the CORE SPARSITY MECHANISM for 2D:
        - At each (S,t), only (degree_S+1)×(degree_T+1) basis functions are non-zero
        - Therefore, σ(S,t) depends on only ~12-16 coefficients (for cubic-quadratic)
        - This creates EXTREMELY SPARSE dependencies in the computational graph

        Args:
            S_points: Array of spot prices to evaluate at
            t: Current time value
            coeff_matrix_advars: 2D array of ADVar coefficients, shape (n_S, n_T)

        Returns:
            List of σ²(S,t) values as ADVars
        """
        sigma_squared_vars = []

        # Get spatial domain boundaries from B-spline model
        S_min = self.bspline_model_2d.S_min
        S_max = self.bspline_model_2d.S_max
        T_min = self.bspline_model_2d.T_min
        T_max = self.bspline_model_2d.T_max

        # Check if t is in extrapolation region
        t_in_extrap = (t < T_min) or (t > T_max)

        # Get temporal basis evaluation method
        if t_in_extrap:
            # Extrapolation: need all basis functions
            basis_T_all = self.bspline_model_2d._evaluate_basis_T(np.array([t]))[0]
            T_indices = list(range(len(basis_T_all)))
            T_basis_vals = list(basis_T_all)
        else:
            # Interpolation: can use active indices safely
            T_indices, T_basis_vals = self._get_active_temporal_basis(t)

        for S in S_points:
            # Check if S is in extrapolation region
            s_in_extrap = (S < S_min) or (S > S_max)

            if s_in_extrap:
                # Extrapolation: need all basis functions
                basis_S_all = self.bspline_model_2d._evaluate_basis_S(np.array([S]))[0]
                S_indices = list(range(len(basis_S_all)))
                S_basis_vals = list(basis_S_all)
            else:
                # Interpolation: can use active indices safely
                S_indices, S_basis_vals = self._get_active_spatial_basis(S)

            # Evaluate σ(S,t) = Σᵢ Σⱼ wᵢⱼ Bᵢ(S) Bⱼ(t)
            sigma_val = ADVar(0.0)

            for i_idx, i in enumerate(S_indices):
                B_S = S_basis_vals[i_idx]
                if abs(B_S) < 1e-14:
                    continue

                for j_idx, j in enumerate(T_indices):
                    B_T = T_basis_vals[j_idx]
                    if abs(B_T) < 1e-14:
                        continue

                    # σ(S,t) += wᵢⱼ * Bᵢ(S) * Bⱼ(t)
                    sigma_val = sigma_val + coeff_matrix_advars[i, j] * ADVar(B_S * B_T)

            # Ensure positivity with floor
            sigma_val_safe = sigma_val + ADVar(1e-6)

            # Compute σ²
            sigma_squared = sigma_val_safe * sigma_val_safe
            sigma_squared_vars.append(sigma_squared)

        return sigma_squared_vars

    def _get_active_spatial_basis(self, S: float) -> Tuple[List[int], List[float]]:
        """
        Get active spatial basis function indices and values at S.

        Returns:
            (active_indices, basis_values) where:
            - active_indices: List of i such that Bᵢ(S) ≠ 0
            - basis_values: List of Bᵢ(S) values
        """
        from scipy.interpolate import BSpline

        active_indices = self.bspline_model_2d.get_active_basis_indices(S, 0.0)[0]
        basis_values = []

        knot_vector_S = self.bspline_model_2d.knot_vector_S
        degree_S = self.bspline_model_2d.degree_S
        n_S = self.bspline_model_2d.coefficients.shape[0]

        for idx in active_indices:
            # Evaluate single basis function Bᵢ(S)
            basis_coeff = np.zeros(n_S)
            basis_coeff[idx] = 1.0
            bspline_basis = BSpline(knot_vector_S, basis_coeff, degree_S, extrapolate=True)
            basis_val = bspline_basis(S)
            basis_values.append(basis_val)

        return active_indices, basis_values

    def _get_active_temporal_basis(self, t: float) -> Tuple[List[int], List[float]]:
        """
        Get active temporal basis function indices and values at t.

        Returns:
            (active_indices, basis_values) where:
            - active_indices: List of j such that Bⱼ(t) ≠ 0
            - basis_values: List of Bⱼ(t) values
        """
        from scipy.interpolate import BSpline

        active_indices = self.bspline_model_2d.get_active_basis_indices(100.0, t)[1]
        basis_values = []

        knot_vector_T = self.bspline_model_2d.knot_vector_T
        degree_T = self.bspline_model_2d.degree_T
        n_T = self.bspline_model_2d.coefficients.shape[1]

        for idx in active_indices:
            # Evaluate single basis function Bⱼ(t)
            basis_coeff = np.zeros(n_T)
            basis_coeff[idx] = 1.0
            bspline_basis = BSpline(knot_vector_T, basis_coeff, degree_T, extrapolate=True)
            basis_val = bspline_basis(t)
            basis_values.append(basis_val)

        return active_indices, basis_values

    def build_tridiagonal_cn_2d(self, coeff_matrix_advars: np.ndarray,
                               dt: ADVar, t: float) -> Tuple:
        """
        Build Crank-Nicolson tridiagonal matrices with 2D B-spline volatility.

        At each grid point (S_i, t), we evaluate σ(S_i, t) = Σⱼ Σₖ wⱼₖ·Bⱼ(S_i)·Bₖ(t).
        For a given time t, the temporal basis is fixed, so we only iterate over
        active temporal basis functions once.

        Args:
            coeff_matrix_advars: 2D array of ADVar coefficients, shape (n_S, n_T)
            dt: Time step (as ADVar)
            t: Current time for volatility evaluation

        Returns:
            (a_L, b_L, c_L, a_R, b_R, c_R): Left and right tridiagonal coefficients
        """
        phi = self.phi
        n = self.M - 2  # Interior points

        # Evaluate σ²(S,t) at all interior grid points using 2D B-spline
        S_interior = self.S_grid[1:-1]
        sigma_squared_vars = self._evaluate_sigma_squared_2d(S_interior, t,
                                                             coeff_matrix_advars)

        # Convert grid spacing to ADVar
        dx_var = ADVar(self.dx, requires_grad=False)
        dx2_var = dx_var * dx_var
        r_var = ADVar(self.r, requires_grad=False)

        a_L, b_L, c_L = [], [], []
        a_R, b_R, c_R = [], [], []

        for i in range(n):
            # Log-space coordinates
            x_i = self.x_grid[1 + i]

            # PDE coefficients in log-space
            sigma2 = sigma_squared_vars[i]

            # Diffusion coefficient
            alpha_i = (sigma2 / ADVar(2.0)) / dx2_var

            # Drift coefficient
            beta_i = (r_var - sigma2 / ADVar(2.0)) / (ADVar(2.0) * dx_var)

            # Tridiagonal entries
            l_i = alpha_i - beta_i  # Lower diagonal
            c_i = -ADVar(2.0) * alpha_i - r_var  # Diagonal (discount term -rV)
            u_i = alpha_i + beta_i  # Upper diagonal

            # Left system (implicit part)
            if i == 0:
                a_L.append(ADVar(0.0))
            else:
                a_L.append(-phi * dt * l_i)

            b_L.append(ADVar(1.0) - phi * dt * c_i)

            if i == n - 1:
                c_L.append(ADVar(0.0))
            else:
                c_L.append(-phi * dt * u_i)

            # Right system (explicit part)
            if i == 0:
                a_R.append(ADVar(0.0))
            else:
                a_R.append(ADVar(1.0 - phi) * dt * l_i)

            b_R.append(ADVar(1.0) + ADVar(1.0 - phi) * dt * c_i)

            if i == n - 1:
                c_R.append(ADVar(0.0))
            else:
                c_R.append(ADVar(1.0 - phi) * dt * u_i)

        return a_L, b_L, c_L, a_R, b_R, c_R

    def tridiag_solve(self, a: List[ADVar], b: List[ADVar], c: List[ADVar],
                     d: List[ADVar]) -> List[ADVar]:
        """Thomas algorithm for tridiagonal system with ADVars."""
        n = len(d)
        c_prime = [None] * n
        d_prime = [None] * n

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i - 1]
            c_prime[i] = c[i] / denom if i < n - 1 else ADVar(0.0)
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

        x = [None] * n
        x[-1] = d_prime[-1]

        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    def cn_step(self, V: List[ADVar], a_L: List[ADVar], b_L: List[ADVar],
                c_L: List[ADVar], a_R: List[ADVar], b_R: List[ADVar],
                c_R: List[ADVar], t_current: float) -> List[ADVar]:
        """Single Crank-Nicolson time step."""
        n = self.M - 2
        rhs = [None] * n

        for i in range(n):
            if i == 0:
                V_left = ADVar(self._boundary_condition_left(t_current), requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i + 1] - a_R[i] * V_left
            elif i == n - 1:
                V_right = ADVar(self._boundary_condition_right(t_current), requires_grad=False)
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] + c_R[i] * V[i + 1]

        V_new = self.tridiag_solve(a_L, b_L, c_L, rhs)
        return V_new

    def _compute_spline_second_derivatives(self, V: List[ADVar],
                                          S_grid: np.ndarray) -> List[ADVar]:
        """
        Compute natural cubic spline second derivatives M_i.

        Natural boundary conditions: M[0] = M[-1] = 0
        """
        n = len(V)

        if n < 3:
            return [ADVar(0.0, requires_grad=False) for _ in range(n)]

        h = np.diff(S_grid)

        # Build tridiagonal system for interior points
        n_interior = n - 2
        A_diag = []
        A_lower = []
        A_upper = []
        rhs = []

        for i in range(1, n - 1):
            h_im1 = h[i - 1]
            h_i = h[i]

            lambda_i = h_im1 / (h_im1 + h_i)
            mu_i = h_i / (h_im1 + h_i)

            # Diagonal is always 2
            A_diag.append(ADVar(2.0))

            # Off-diagonals
            if i > 1:
                A_lower.append(ADVar(lambda_i))
            if i < n - 2:
                A_upper.append(ADVar(mu_i))

            # RHS
            d_i = (ADVar(6.0) / ADVar(h_im1 + h_i)) * (
                (V[i + 1] - V[i]) / ADVar(h_i) - (V[i] - V[i - 1]) / ADVar(h_im1)
            )
            rhs.append(d_i)

        if n_interior == 0:
            return [ADVar(0.0)] * n
        elif n_interior == 1:
            M_interior = [rhs[0] / A_diag[0]]
        else:
            # Solve tridiagonal system
            a = [ADVar(0.0)] + A_lower
            b = A_diag
            c = A_upper + [ADVar(0.0)]

            M_interior = self.tridiag_solve(a, b, c, rhs)

        # Add natural boundary conditions
        M_vals = [ADVar(0.0)] + M_interior + [ADVar(0.0)]

        return M_vals

    def solve_pde_with_aad(self, S0_val: float, coeff_matrix_vals: np.ndarray,
                          compute_hessian: bool = True, verbose: bool = False) -> Dict:
        """
        Solve BS PDE with 2D B-spline volatility using AAD edge-pushing.

        Args:
            S0_val: Initial stock price to evaluate option at
            coeff_matrix_vals: 2D B-spline coefficient matrix, shape (n_S, n_T)
                              Can also be flattened 1D array
            compute_hessian: Whether to compute full Hessian
            verbose: Print diagnostic information

        Returns:
            Dictionary with:
                - price: Option price
                - jacobian: ∂V/∂wᵢⱼ (gradient vector, flattened)
                - hessian: ∂²V/∂wᵢⱼ∂wₖₗ (Hessian matrix, flattened indices)
                - sparsity_info: Sparsity statistics
                - computation_time_ms: Wall-clock time
        """
        from aad_edge_pushing.edge_pushing.algo4_cython_simple import algo4_cython_simple

        t_start = time.perf_counter()
        global_tape.reset()

        # Handle both flat and 2D input
        coeff_matrix_vals = np.asarray(coeff_matrix_vals, dtype=np.float64)
        if coeff_matrix_vals.ndim == 1:
            expected_shape = self.bspline_model_2d.coefficients.shape
            coeff_matrix_vals = coeff_matrix_vals.reshape(expected_shape)

        n_S, n_T = coeff_matrix_vals.shape
        n_params = n_S * n_T

        # Get active parameters for sparse optimization
        active_params = self.get_active_parameters()
        n_active = len(active_params)

        # Create ADVars for all 2D B-spline coefficients
        # Only active parameters get requires_grad=True (sparse optimization)
        coeff_matrix_advars = np.empty((n_S, n_T), dtype=object)
        coeff_advars_flat = []  # For edge-pushing (flattened list)

        for i in range(n_S):
            for j in range(n_T):
                k = i * n_T + j  # Flat index
                is_active = (i, j) in active_params
                advar = ADVar(coeff_matrix_vals[i, j], requires_grad=is_active,
                             name=f"w{i},{j}")
                coeff_matrix_advars[i, j] = advar
                coeff_advars_flat.append(advar)

        # Time grid
        N = self.N
        dt_val = self.dt
        dt = ADVar(dt_val, requires_grad=False)

        if verbose:
            print(f"  PDE Grid: M={self.M}, N={N}")
            print(f"  Active parameters: {n_active}/{n_params} ({100*n_active/n_params:.1f}%)")
            print(f"  2D B-spline: {n_S}×{n_T} = {n_params} coefficients")
            print(f"  Degrees: (S={self.bspline_model_2d.degree_S}, "
                  f"T={self.bspline_model_2d.degree_T})")
            print(f"  Time step: dt={dt_val:.6f}, dx={self.dx:.4f}")

        # Terminal condition
        V_terminal = self._terminal_condition()
        V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

        # Time stepping (backward in time)
        for n in range(N):
            t_current = self.t_grid[n + 1]  # Current time for volatility evaluation

            # Build tridiagonal matrices for this time step
            # Volatility σ(S, t_current) varies with time!
            a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_cn_2d(
                coeff_matrix_advars, dt, t_current
            )

            V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_current)

        # Natural cubic spline interpolation to get V(S0)
        S_interior = self.S_grid[1:-1]
        M_vals = self._compute_spline_second_derivatives(V, S_interior)

        # Find interval containing S0
        idx = np.searchsorted(S_interior, S0_val)
        if idx == 0:
            idx = 1
        elif idx >= len(S_interior):
            idx = len(S_interior) - 1

        i = idx - 1
        S_i = S_interior[i]
        S_i1 = S_interior[i + 1]
        V_i = V[i]
        V_i1 = V[i + 1]
        M_i = M_vals[i]
        M_i1 = M_vals[i + 1]

        h = S_i1 - S_i

        # Cubic spline evaluation at S0_val
        S0_const = S0_val
        A_val = (S_i1 - S0_const) / h
        B_val = (S0_const - S_i) / h

        A = ADVar(A_val, requires_grad=False)
        B = ADVar(B_val, requires_grad=False)
        A3 = A * A * A
        B3 = B * B * B

        h_var = ADVar(h, requires_grad=False)
        h2_over_6 = h_var * h_var / ADVar(6.0)

        price_var = (A * V_i + B * V_i1 +
                    (A3 - A) * h2_over_6 * M_i +
                    (B3 - B) * h2_over_6 * M_i1)

        # Extract price
        price = price_var.val

        # Compute Hessian using edge-pushing
        if verbose:
            print(f"  Computing Hessian via edge-pushing (algo4)...")

        hessian = algo4_cython_simple(price_var, coeff_advars_flat)

        # Compute gradient (Jacobian)
        for var in coeff_advars_flat:
            var.adj = 0.0

        price_var.adj = 1.0

        # Reverse mode traversal
        for node in reversed(global_tape.nodes):
            if node.out.adj != 0:
                for parent, partial in node.parents:
                    parent.adj += node.out.adj * partial

        # Collect gradients
        gradient = np.array([var.adj for var in coeff_advars_flat])

        # Sparsity analysis
        sparsity_pattern = self.bspline_model_2d.get_hessian_sparsity_pattern()
        bandwidth_2d = self.bspline_model_2d.get_hessian_bandwidth_2d()

        # Actual sparsity from computed Hessian
        threshold = 1e-10
        nonzero_mask = np.abs(hessian) > threshold
        n_nonzero_actual = np.sum(nonzero_mask)
        n_total = n_params * n_params
        sparsity_actual = n_nonzero_actual / n_total

        # Compare with theoretical pattern
        matches_theory = np.allclose(nonzero_mask, sparsity_pattern)

        computation_time_ms = (time.perf_counter() - t_start) * 1000

        if verbose:
            print(f"  Price: {price:.6f}")
            print(f"  Gradient norm: {np.linalg.norm(gradient):.6e}")
            print(f"  Hessian sparsity: {100*(1-sparsity_actual):.1f}% zeros")
            print(f"  2D bandwidth: {bandwidth_2d}")
            print(f"  Matches theory: {matches_theory}")
            print(f"  Computation time: {computation_time_ms:.2f} ms")

        return {
            'price': price,
            'jacobian': gradient,
            'hessian': hessian,
            'sparsity_info': {
                'n_params': n_params,
                'n_params_2d': (n_S, n_T),
                'bandwidth_2d': bandwidth_2d,
                'actual_nonzero': n_nonzero_actual,
                'sparsity_ratio': sparsity_actual,
                'zero_ratio': 1.0 - sparsity_actual,
                'matches_theory': matches_theory,
                'theoretical_pattern': sparsity_pattern,
                'actual_pattern': nonzero_mask
            },
            'computation_time_ms': computation_time_ms,
            'n_pde_solves': 1,
            'tape_size': len(global_tape.nodes)
        }


if __name__ == "__main__":
    from aad_edge_pushing.pde.models.bspline_model_2d import (
        BSplineModel2D, BSplineConfig2D, create_flat_bspline_2d
    )

    print("="*70)
    print("2D B-Spline PDE Solver Test")
    print("="*70)

    # Create flat 2D B-spline (should match 1D constant vol)
    print("\n1. Creating flat 2D B-spline model...")
    config = BSplineConfig2D(
        S_min=50.0, S_max=150.0, T_min=0.0, T_max=1.0,
        n_knots_S=5, n_knots_T=3,
        degree_S=3, degree_T=2
    )
    model_2d = create_flat_bspline_2d(config, volatility=0.25)
    print(f"   Model: {model_2d.coefficients.shape[0]}×{model_2d.coefficients.shape[1]} "
          f"= {model_2d.get_n_params()} parameters")

    # Create PDE solver
    print("\n2. Creating PDE solver...")
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    solver = BS_PDE_AAD_BSpline2D(
        model_2d, S0, K, T, r,
        M=100, N_base=25,
        use_adaptive_Smax=True
    )
    print(f"   Grid: M={solver.M}, N={solver.N}")
    print(f"   S_max: {solver.S_max:.2f}")
    print(f"   CFL ratio: {solver.cfl_ratio:.3f}")

    # Solve PDE
    print("\n3. Solving PDE with edge-pushing...")
    coeff_vals = model_2d.get_coefficients_2d()
    result = solver.solve_pde_with_aad(S0, coeff_vals, verbose=True)

    print("\n4. Results:")
    print(f"   Price: {result['price']:.6f}")
    print(f"   Gradient shape: {result['jacobian'].shape}")
    print(f"   Hessian shape: {result['hessian'].shape}")
    print(f"   Sparsity: {result['sparsity_info']['zero_ratio']:.1%} zeros")
    print(f"   Time: {result['computation_time_ms']:.2f} ms")
    print(f"   Tape size: {result['tape_size']} nodes")

    # Compare with Black-Scholes analytical
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r + 0.5*0.25**2)*T) / (0.25*np.sqrt(T))
    d2 = d1 - 0.25*np.sqrt(T)
    bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

    print(f"\n5. Validation:")
    print(f"   PDE price: {result['price']:.6f}")
    print(f"   BS analytical: {bs_price:.6f}")
    print(f"   Relative error: {abs(result['price'] - bs_price)/bs_price * 100:.3f}%")

    print("\n" + "="*70)
    print("✓ 2D B-spline PDE solver test completed!")
    print("="*70)
