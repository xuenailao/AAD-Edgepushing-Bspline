"""
Pure PDE Solver for B-spline Volatility (No AAD)

This is a lightweight PDE solver specifically for Bumping2 finite difference
Hessian computation. It does NOT use automatic differentiation.

Key differences from pde_aad_bspline.py:
    - Uses plain floats instead of ADVar
    - No tape recording
    - No computational graph
    - ~7-10× faster for single price evaluation
    - Perfect for parallel Bumping2

Author: AAD Research
Date: 2025-11-15
"""

import numpy as np
from typing import List, Dict
import time
from scipy.interpolate import BSpline


class BS_PDE_Pure_BSpline:
    """
    Black-Scholes PDE solver with B-spline volatility (pure numerical, no AAD).

    Solves: ∂V/∂t + (r - σ²/2) * S * ∂V/∂S + σ²/2 * S² * ∂²V/∂S² - r*V = 0

    where σ(S) is represented by a B-spline model.
    """

    def __init__(self, bspline_model, S0: float, K: float, T: float, r: float,
                 M: int = 151, N_base: int = 50, n_rannacher: int = 4,
                 use_adaptive_Smax: bool = True, sigma_margin: float = 0.1):
        """
        Initialize pure PDE solver.

        Parameters
        ----------
        bspline_model : BSplineModel
            B-spline volatility model
        S0, K, T, r : float
            Market parameters
        M : int
            Number of spatial grid points
        N_base : int
            Base number of time steps
        n_rannacher : int
            Number of Rannacher smoothing steps
        use_adaptive_Smax : bool
            Use adaptive S_max based on volatility
        sigma_margin : float
            Margin for adaptive S_max computation
        """
        self.bspline_model = bspline_model
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.M = M
        self.N_base = N_base
        self.n_rannacher = n_rannacher
        self.use_adaptive_Smax = use_adaptive_Smax
        self.sigma_margin = sigma_margin

        # Setup grids
        self._setup_grids()

    def _setup_grids(self):
        """Setup spatial and temporal grids."""
        from aad_edge_pushing.pde.pde_config import PDEConfig

        # Spatial grid
        S_min = 1e-3

        if self.use_adaptive_Smax:
            # Estimate typical volatility
            sigma_typical = self.bspline_model.evaluate(np.array([self.S0]))[0]
            S_max = PDEConfig.compute_unified_Smax(
                self.K, self.T, sigma_typical, self.r, self.sigma_margin
            )
        else:
            S_max = 5.0 * self.K

        # Use log-space grid (same as AAD solver)
        self.x_grid, self.S_grid, self.dx = PDEConfig.create_log_space_grid(S_min, S_max, self.M)

        # Time grid (with Rannacher smoothing)
        self.N = self.N_base + self.n_rannacher
        self.t_grid = np.linspace(0, self.T, self.N + 1)
        self.dt = self.t_grid[1] - self.t_grid[0]

    def solve(self, S0_val: float, coeff_vals: np.ndarray) -> float:
        """
        Solve PDE and return option price at S0_val.

        This is a PURE numerical method - no AAD, no tape, just PDE solving.

        Parameters
        ----------
        S0_val : float
            Spot price to evaluate at
        coeff_vals : np.ndarray
            B-spline coefficient values

        Returns
        -------
        float
            Option price V(S0_val, t=0)
        """
        # Update B-spline coefficients
        self.bspline_model.coefficients = coeff_vals.copy()

        # IMPORTANT: Rebuild the internal BSpline object
        # (scipy's BSpline doesn't update when coefficients are changed)
        from scipy.interpolate import BSpline
        self.bspline_model._bspline = BSpline(
            self.bspline_model.knot_vector,
            self.bspline_model.coefficients,
            self.bspline_model.degree,
            extrapolate=True
        )

        # Evaluate σ²(S) at interior grid points (plain floats)
        S_interior = self.S_grid[1:-1]
        sigma_squared = self._evaluate_sigma_squared(S_interior)

        # Build base tridiagonal coefficients (plain floats)
        a_base, b_base, c_base = self._build_base_tridiagonal(sigma_squared)

        # Terminal condition
        V_terminal = self._terminal_condition()
        V = V_terminal[1:-1].copy()  # Interior points

        # Time stepping (backward in time)
        for n in range(self.N):
            t_current = self.t_grid[n + 1]

            # Determine theta (Rannacher smoothing)
            if n < self.n_rannacher:
                theta = 1.0  # Fully implicit
            else:
                theta = 0.5  # Crank-Nicolson

            V = self._cn_step(V, a_base, b_base, c_base, t_current, theta)

        # Interpolate to S0_val using cubic spline
        price = self._interpolate_cubic(V, S_interior, S0_val)

        return price

    def _evaluate_sigma_squared(self, S_points: np.ndarray) -> np.ndarray:
        """
        Evaluate σ²(S) using B-spline (plain floats).

        Parameters
        ----------
        S_points : np.ndarray
            Points to evaluate at

        Returns
        -------
        np.ndarray
            σ²(S) values (plain floats)
        """
        # Use BSplineModel's evaluate method
        sigma_vals = self.bspline_model.evaluate(S_points)

        # Ensure positivity
        sigma_vals = np.maximum(sigma_vals, 1e-6)

        return sigma_vals ** 2

    def _build_base_tridiagonal(self, sigma_squared: np.ndarray):
        """
        Build base tridiagonal coefficients (l_i, c_i, u_i) for PDE discretization.

        These are the BASE coefficients of the spatial operator L, WITHOUT any
        theta-method or time-stepping factors. The theta factors are applied
        dynamically in _cn_step to support Rannacher smoothing.

        Returns
        -------
        Tuple of (a_base, b_base, c_base)
            Base lower/diagonal/upper coefficients
        """
        n = self.M - 2
        dx = self.dx
        dx2 = dx * dx
        r = self.r

        a_base = np.zeros(n)  # Lower diagonal
        b_base = np.zeros(n)  # Diagonal
        c_base = np.zeros(n)  # Upper diagonal

        for i in range(n):
            sigma2 = sigma_squared[i]

            # Diffusion coefficient
            alpha_i = (sigma2 / 2.0) / dx2

            # Drift coefficient (in log-space)
            beta_i = (r - sigma2 / 2.0) / dx

            # Tridiagonal entries for spatial operator L
            l_i = alpha_i - beta_i / 2.0  # Lower diagonal
            c_i = -2.0 * alpha_i + r       # Diagonal
            u_i = alpha_i + beta_i / 2.0   # Upper diagonal

            # Store base coefficients
            if i > 0:
                a_base[i] = l_i
            b_base[i] = c_i
            if i < n-1:
                c_base[i] = u_i

        return a_base, b_base, c_base

    def _cn_step(self, V: np.ndarray, a_base: np.ndarray, b_base: np.ndarray,
                 c_base: np.ndarray, t_current: float, theta: float = 0.5) -> np.ndarray:
        """
        Single Crank-Nicolson/fully implicit time step (plain floats).

        NOTE: a_base, b_base, c_base are the BASE tridiagonal coefficients (l_i, c_i, u_i)
        WITHOUT theta-method factors. We apply theta here dynamically.

        Parameters
        ----------
        V : np.ndarray
            Current solution
        a_base, b_base, c_base : np.ndarray
            Base tridiagonal coefficients
        t_current : float
            Current time
        theta : float
            Implicitness parameter (0.5 = CN, 1.0 = fully implicit)

        Returns
        -------
        np.ndarray
            Updated solution
        """
        n = len(V)
        dt = self.dt

        # Build left system: (I - theta*dt*L) * V^{n+1}
        a_left = np.zeros(n)
        b_left = np.zeros(n)
        c_left = np.zeros(n)

        for i in range(n):
            if i > 0:
                a_left[i] = -theta * dt * a_base[i]
            b_left[i] = 1.0 - theta * dt * b_base[i]
            if i < n-1:
                c_left[i] = -theta * dt * c_base[i]

        # Build right side: (I + (1-theta)*dt*L) * V^n + BC
        rhs = np.zeros(n)

        for i in range(n):
            if i == 0:
                V_left = self._boundary_condition_left(t_current)
                rhs[i] = ((1.0 + (1.0-theta) * dt * b_base[i]) * V[i] +
                         (1.0-theta) * dt * c_base[i] * V[i+1] -
                         (1.0-theta) * dt * a_base[i] * V_left)
            elif i == n-1:
                V_right = self._boundary_condition_right(t_current)
                rhs[i] = ((1.0-theta) * dt * a_base[i] * V[i-1] +
                         (1.0 + (1.0-theta) * dt * b_base[i]) * V[i] -
                         (1.0-theta) * dt * c_base[i] * V_right)
            else:
                rhs[i] = ((1.0-theta) * dt * a_base[i] * V[i-1] +
                         (1.0 + (1.0-theta) * dt * b_base[i]) * V[i] +
                         (1.0-theta) * dt * c_base[i] * V[i+1])

        # Solve tridiagonal system
        V_new = self._tridiag_solve(a_left, b_left, c_left, rhs)

        return V_new

    def _tridiag_solve(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                       d: np.ndarray) -> np.ndarray:
        """
        Thomas algorithm for tridiagonal system (plain floats).

        Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
        """
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            if i < n-1:
                c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        x = np.zeros(n)
        x[-1] = d_prime[-1]

        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def _terminal_condition(self) -> np.ndarray:
        """Call option payoff at maturity."""
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_condition_left(self, t: float) -> float:
        """Boundary condition at S=S_min ≈ 0."""
        return 0.0

    def _boundary_condition_right(self, t: float) -> float:
        """Boundary condition at S=S_max (large S)."""
        tau = self.T - t
        return self.S_grid[-1] - self.K * np.exp(-self.r * tau)

    def _interpolate_cubic(self, V: np.ndarray, S_interior: np.ndarray,
                           S0_val: float) -> float:
        """
        Natural cubic spline interpolation to get V(S0_val).

        Parameters
        ----------
        V : np.ndarray
            Solution at interior grid points
        S_interior : np.ndarray
            Interior grid points
        S0_val : float
            Point to interpolate at

        Returns
        -------
        float
            Interpolated value
        """
        # Compute natural cubic spline second derivatives
        M_vals = self._compute_spline_second_derivatives(V, S_interior)

        # Find interval containing S0_val
        idx = np.searchsorted(S_interior, S0_val)
        if idx == 0:
            idx = 1
        elif idx >= len(S_interior):
            idx = len(S_interior) - 1

        i = idx - 1
        S_i = S_interior[i]
        S_i1 = S_interior[i+1]
        V_i = V[i]
        V_i1 = V[i+1]
        M_i = M_vals[i]
        M_i1 = M_vals[i+1]

        h = S_i1 - S_i

        # Cubic spline formula
        A = (S_i1 - S0_val) / h
        B = (S0_val - S_i) / h

        price = (A * V_i + B * V_i1 +
                (A**3 - A) * h**2 / 6.0 * M_i +
                (B**3 - B) * h**2 / 6.0 * M_i1)

        return price

    def _compute_spline_second_derivatives(self, V: np.ndarray,
                                          S_grid: np.ndarray) -> np.ndarray:
        """
        Compute natural cubic spline second derivatives M_i.

        Natural boundary conditions: M[0] = M[-1] = 0
        """
        n = len(V)

        if n < 3:
            return np.zeros(n)

        h = np.diff(S_grid)

        # Build tridiagonal system for interior points
        n_interior = n - 2

        if n_interior == 0:
            return np.zeros(n)

        # Tridiagonal matrix: A_lower, A_diag, A_upper
        # For n_interior unknowns, need:
        #   A_diag: size n_interior
        #   A_lower: size n_interior (first element unused)
        #   A_upper: size n_interior (last element unused)
        A_diag = np.zeros(n_interior)
        A_lower = np.zeros(n_interior)
        A_upper = np.zeros(n_interior)
        rhs = np.zeros(n_interior)

        for i in range(1, n-1):
            idx = i - 1  # Index into interior arrays
            h_im1 = h[i-1]
            h_i = h[i]

            lambda_i = h_im1 / (h_im1 + h_i)
            mu_i = h_i / (h_im1 + h_i)

            # Diagonal is always 2
            A_diag[idx] = 2.0

            # Lower diagonal (except first row)
            if idx > 0:
                A_lower[idx] = lambda_i

            # Upper diagonal (except last row)
            if idx < n_interior - 1:
                A_upper[idx] = mu_i

            # RHS
            d_i = 6.0 / (h_im1 + h_i) * ((V[i+1] - V[i])/h_i - (V[i] - V[i-1])/h_im1)
            rhs[idx] = d_i

        # Solve tridiagonal system
        M_interior = self._tridiag_solve(A_lower, A_diag, A_upper, rhs)

        # Add natural boundary conditions
        M = np.zeros(n)
        M[1:-1] = M_interior

        return M


def create_pure_solver(bspline_model, S0: float, K: float, T: float, r: float,
                       M: int = 151, N: int = 50,
                       use_adaptive_Smax: bool = True,
                       sigma_margin: float = 0.1) -> BS_PDE_Pure_BSpline:
    """
    Convenience function to create pure PDE solver.

    Parameters
    ----------
    bspline_model : BSplineModel
        B-spline volatility model
    S0, K, T, r : float
        Market parameters
    M, N : int
        Grid resolution
    use_adaptive_Smax : bool
        Use adaptive S_max
    sigma_margin : float
        Margin for adaptive S_max

    Returns
    -------
    BS_PDE_Pure_BSpline
        Pure PDE solver instance
    """
    return BS_PDE_Pure_BSpline(
        bspline_model=bspline_model,
        S0=S0, K=K, T=T, r=r,
        M=M, N_base=N,
        use_adaptive_Smax=use_adaptive_Smax,
        sigma_margin=sigma_margin
    )
