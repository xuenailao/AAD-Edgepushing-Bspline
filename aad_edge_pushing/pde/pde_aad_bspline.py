"""
Black-Scholes PDE Solver with B-Spline Volatility Surface and AAD Edge-Pushing

This module implements a complete PDE solver where the volatility is parameterized
as a B-spline function σ(S) = Σᵢ wᵢ·Bᵢ(S), with all coefficients {wᵢ} as ADVars.

Key Features:
- B-spline volatility: σ(S) computed locally at each grid point
- Sparse Hessian: Edge-pushing exploits compact support
- Natural cubic spline interpolation for final price extraction
- Crank-Nicolson time stepping with ADVar tridiagonal solver

Integration:
This extends the existing BS_PDE_AAD to handle vector parameters (B-spline coeffs)
instead of scalar parameter (constant sigma).
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


class BS_PDE_AAD_BSpline:
    """
    Black-Scholes PDE solver with B-spline volatility parameterization.

    The volatility σ(S) = Σᵢ wᵢ·Bᵢ(S) is evaluated locally at each grid point.
    All coefficients wᵢ are ADVars, enabling edge-pushing to compute the full
    Hessian matrix ∂²V/∂wᵢ∂wⱼ in a single backward pass.
    """

    def __init__(self, bspline_model, S0: float, K: float, T: float, r: float,
                 M: int = 151, N_base: int = 50,
                 use_adaptive_Smax: bool = True, sigma_margin: float = 0.1):
        """
        Initialize BS PDE solver with B-spline volatility.

        Args:
            bspline_model: BSplineModel instance (defines knots, degree, domain)
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            M: Number of spatial grid points
            N_base: Base number of time steps
            n_rannacher: Number of Rannacher smoothing steps (not implemented in AAD)
            use_adaptive_Smax: Use volatility-adaptive S_max
            sigma_margin: Safety margin for grid sizing
        """
        from aad_edge_pushing.pde.models import BSplineModel

        if not isinstance(bspline_model, BSplineModel):
            raise TypeError("bspline_model must be a BSplineModel instance")

        self.bspline_model = bspline_model
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.M = M
        self.N_base = N_base

        # Estimate typical volatility for grid sizing
        # Use volatility at S0 as representative
        sigma_typical = bspline_model.evaluate(np.array([S0]))[0]

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

        # CFL analysis using typical volatility
        alpha_typical = (sigma_typical**2 / 2.0) / (self.dx**2)
        dt_critical = (self.dx**2) / (2.0 * alpha_typical)
        self.cfl_ratio = self.dt / dt_critical

        # Crank-Nicolson parameter (Rannacher not implemented for AAD)
        self.phi = 0.5

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

    def build_tridiagonal_cn(self, coeff_advars: List[ADVar], dt: ADVar) -> Tuple:
        """
        Build Crank-Nicolson tridiagonal matrices with B-spline volatility.

        At each grid point S_i, we evaluate σ(S_i) = Σⱼ wⱼ·Bⱼ(S_i).
        Only (degree+1) coefficients are active at each point → sparse dependencies!

        Args:
            coeff_advars: List of B-spline coefficient ADVars [w₀, w₁, ..., wₙ]
            dt: Time step (as ADVar)

        Returns:
            (a_L, b_L, c_L, a_R, b_R, c_R): Left and right tridiagonal coefficients
        """
        phi = self.phi
        n = self.M - 2  # Interior points

        # Evaluate σ²(S) at all interior grid points using B-spline
        S_interior = self.S_grid[1:-1]
        sigma_squared_vars = self._evaluate_sigma_squared(S_interior, coeff_advars)

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
            # (∂V/∂τ) = (σ²/2) * ∂²V/∂x² + (r - σ²/2) * ∂V/∂x - r*V
            sigma2 = sigma_squared_vars[i]

            # Diffusion coefficient
            alpha_i = (sigma2 / ADVar(2.0)) / dx2_var

            # Drift coefficient
            beta_i = (r_var - sigma2 / ADVar(2.0)) / (ADVar(2.0) * dx_var)

            # Tridiagonal entries
            l_i = alpha_i - beta_i  # Lower diagonal
            c_i = -ADVar(2.0) * alpha_i - r_var     # Diagonal (discount term -rV)
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

    def _evaluate_sigma_squared(self, S_points: np.ndarray,
                                coeff_advars: List[ADVar]) -> List[ADVar]:
        """
        Evaluate σ²(S) at given points using B-spline with ADVar coefficients.

        This is the CORE SPARSITY MECHANISM:
        - At each S, only (degree+1) basis functions are non-zero
        - Therefore, σ(S) depends on only (degree+1) coefficients
        - This creates SPARSE dependencies in the computational graph

        Args:
            S_points: Array of spot prices to evaluate at
            coeff_advars: List of B-spline coefficient ADVars

        Returns:
            List of σ²(S) values as ADVars
        """
        from scipy.interpolate import BSpline

        sigma_squared_vars = []

        # Get B-spline basis information
        knot_vector = self.bspline_model.knot_vector
        degree = self.bspline_model.degree

        for S in S_points:
            # Find active basis functions at this S
            active_indices = self.bspline_model.get_support_indices(S)

            # Evaluate basis functions at S (numerical values)
            sigma_val = ADVar(0.0)

            for idx in active_indices:
                # Evaluate single basis function Bᵢ(S)
                basis_coeff = np.zeros(len(coeff_advars))
                basis_coeff[idx] = 1.0
                bspline_basis = BSpline(knot_vector, basis_coeff, degree)
                basis_value = bspline_basis(S)

                if abs(basis_value) > 1e-14:
                    # σ(S) += wᵢ * Bᵢ(S)
                    sigma_val = sigma_val + coeff_advars[idx] * ADVar(basis_value)

            # Ensure positivity (volatility must be positive)
            # Note: coefficients should already be positive, but we add a floor
            sigma_val_safe = sigma_val + ADVar(1e-6)  # Small floor

            # Compute σ²
            sigma_squared = sigma_val_safe * sigma_val_safe
            sigma_squared_vars.append(sigma_squared)

        return sigma_squared_vars

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
            # Only boundary points
            return [ADVar(0.0)] * n
        elif n_interior == 1:
            # Single interior point
            M_interior = [rhs[0] / A_diag[0]]
        else:
            # Solve tridiagonal system
            # Convert to standard form for tridiag_solve
            a = [ADVar(0.0)] + A_lower
            b = A_diag
            c = A_upper + [ADVar(0.0)]

            M_interior = self.tridiag_solve(a, b, c, rhs)

        # Add natural boundary conditions
        M_vals = [ADVar(0.0)] + M_interior + [ADVar(0.0)]

        return M_vals

    def solve_pde_with_aad(self, S0_val: float, coeff_vals: List[float],
                          compute_hessian: bool = True, verbose: bool = False) -> Dict:
        """
        Solve BS PDE with B-spline volatility using AAD edge-pushing.

        Args:
            S0_val: Initial stock price to evaluate option at
            coeff_vals: B-spline coefficient values [w₀, w₁, ..., wₙ]
            compute_hessian: Whether to compute full Hessian (always True for B-spline)
            verbose: Print diagnostic information

        Returns:
            Dictionary with:
                - price: Option price
                - jacobian: ∂V/∂wᵢ (gradient vector)
                - hessian: ∂²V/∂wᵢ∂wⱼ (Hessian matrix)
                - sparsity_info: Sparsity statistics
                - computation_time_ms: Wall-clock time
        """
        from aad_edge_pushing.edge_pushing.algo4_cython_simple import algo4_cython_simple

        t_start = time.perf_counter()
        global_tape.reset()

        # Create ADVars for all B-spline coefficients
        n_params = len(coeff_vals)
        coeff_advars = [ADVar(val, requires_grad=True, name=f"w{i}")
                       for i, val in enumerate(coeff_vals)]

        # Time grid
        N = self.N
        t_grid = np.linspace(0, self.T, N + 1)
        dt_val = t_grid[1] - t_grid[0]
        dt = ADVar(dt_val, requires_grad=False)

        if verbose:
            print(f"  PDE Grid: M={self.M}, N={N}")
            print(f"  B-spline: {n_params} coefficients (degree={self.bspline_model.degree})")
            print(f"  Time step: dt={dt_val:.6f}, dx={self.dx:.4f}")

        # Build tridiagonal matrices (depends on ALL active coefficients)
        a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_cn(coeff_advars, dt)

        # Terminal condition
        V_terminal = self._terminal_condition()
        V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

        # Time stepping (backward in time)
        for n in range(N):
            t_current = t_grid[n + 1]
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

        # Cubic spline evaluation at S0_val (constant, not ADVar since we're not differentiating wrt S0)
        S0_const = S0_val  # Numerical value
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

        hessian = algo4_cython_simple(price_var, coeff_advars)

        # Compute gradient (Jacobian)
        # Initialize adjoints
        for var in coeff_advars:
            var.adj = 0.0

        price_var.adj = 1.0

        # Reverse mode traversal
        for node in reversed(global_tape.nodes):
            if node.out.adj != 0:
                for parent, partial in node.parents:
                    parent.adj += node.out.adj * partial

        # Collect gradients
        gradient = np.array([var.adj for var in coeff_advars])

        # Sparsity analysis
        sparsity_pattern = self.bspline_model.get_hessian_sparsity_pattern()
        expected_bandwidth = self.bspline_model.get_hessian_bandwidth()

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
            print(f"  Matches theory: {matches_theory}")
            print(f"  Computation time: {computation_time_ms:.2f} ms")

        return {
            'price': price,
            'jacobian': gradient,
            'hessian': hessian,
            'sparsity_info': {
                'n_params': n_params,
                'expected_bandwidth': expected_bandwidth,
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
