"""
Simple PDE Solver for Black-Scholes Equation
Pure numerical solver without AAD - for use in finite difference methods

Enhancements:
- Adaptive S_max based on volatility
- Rannacher time smoothing for payoff discontinuities
- Unified grid interpolation for bumping consistency
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import CubicSpline

from .pde_config import PDEConfig


class SimplePDESolver:
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 M: int = 151, N_base: int = 50, n_rannacher: int = 4,
                 use_adaptive_Smax: bool = True, sigma_margin: float = 0.1,
                 S_max_override: Optional[float] = None,
                 volatility_model=None):
        """
        Initialize PDE solver for Black-Scholes equation

        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility (constant, or representative for grid sizing)
            M: Number of spatial grid points
            N_base: Number of time steps
            n_rannacher: Number of Rannacher smoothing steps (0 to disable)
            use_adaptive_Smax: If True, S_max adapts to sigma; if False, S_max = 5K
            sigma_margin: Safety margin for bumping (used when use_adaptive_Smax=True)
            S_max_override: If provided, use this S_max instead of computing it
            volatility_model: Optional B-spline model for σ(S) or σ(S,t)
                            If provided, overrides constant sigma
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N_base = N_base
        self.n_rannacher = n_rannacher
        self.volatility_model = volatility_model

        # Adaptive S_max with margin for bumping consistency
        if S_max_override is not None:
            # Use externally provided S_max (for unified bumping)
            S_max = S_max_override
        elif use_adaptive_Smax:
            S_max = PDEConfig.compute_unified_Smax(K, T, sigma, r, sigma_margin)
        else:
            # Fallback to fixed S_max for backward compatibility
            S_max = 5.0 * K

        self.S_max = S_max

        # Create log-space grid
        S_min = 1e-3
        self.x_grid, self.S_grid, self.dx = PDEConfig.create_log_space_grid(S_min, S_max, M)

    def _terminal_condition(self) -> np.ndarray:
        """Terminal payoff: max(S - K, 0) for call option"""
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_condition_left(self, t: float) -> float:
        """Boundary condition at S=0: option worth 0"""
        return 0.0

    def _boundary_condition_right(self, t: float) -> float:
        """Boundary condition at S=S_max: option worth S - K*exp(-r*(T-t))"""
        T_remain = self.T - t
        return self.S_grid[-1] - self.K * np.exp(-self.r * T_remain)

    def _solve_pde_numerical(self, S0: float, sigma: float) -> Tuple[float, np.ndarray]:
        """
        Solve PDE and return price at S0

        Args:
            S0: Stock price to evaluate
            sigma: Volatility (can differ from self.sigma for bumping)
                  Ignored if volatility_model is provided

        Returns:
            price: Option value at S0
            V_interior: Solution on interior grid points
        """
        N = self.N_base
        dt = self.T / N
        t_grid = np.linspace(0, self.T, N + 1)

        n = self.M - 2  # Interior points
        dx = self.dx

        # Terminal condition
        V_terminal = self._terminal_condition()
        V = V_terminal[1:-1].copy()

        # Time stepping with Rannacher smoothing
        for n_step in range(N):
            # Dynamic phi: fully implicit for first n_rannacher steps
            phi = PDEConfig.get_rannacher_phi(n_step, self.n_rannacher)

            t_current = t_grid[n_step+1]
            V_left = self._boundary_condition_left(t_current)
            V_right = self._boundary_condition_right(t_current)

            # Get volatility for this time step
            if self.volatility_model is not None:
                # Time-varying volatility: evaluate σ(S,t) at interior points
                S_interior = self.S_grid[1:-1]
                t_current_arr = np.full_like(S_interior, t_current)
                sigma_grid = self.volatility_model.evaluate(S_interior, t_current_arr)
            else:
                # Constant volatility
                sigma_grid = np.full(n, sigma)

            # Build coefficient matrices (now spatially varying if model provided)
            a_L = np.zeros(n)
            b_L = np.zeros(n)
            c_L = np.zeros(n)
            a_R = np.zeros(n)
            b_R = np.zeros(n)
            c_R = np.zeros(n)

            for i in range(n):
                # PDE coefficients in log-space at grid point i
                sigma_i = sigma_grid[i]
                alpha = 0.5 * sigma_i**2 / (dx**2)  # Diffusion
                beta = (self.r - 0.5 * sigma_i**2) / (2.0 * dx)  # Drift
                gamma = -self.r  # Discount

                l = alpha - beta  # Lower diagonal
                c = -2.0 * alpha + gamma  # Main diagonal
                u = alpha + beta  # Upper diagonal

                a_L[i] = -phi * dt * l if i > 0 else 0.0
                b_L[i] = 1.0 - phi * dt * c
                c_L[i] = -phi * dt * u if i < n-1 else 0.0

                a_R[i] = (1.0 - phi) * dt * l if i > 0 else 0.0
                b_R[i] = 1.0 + (1.0 - phi) * dt * c
                c_R[i] = (1.0 - phi) * dt * u if i < n-1 else 0.0

            # Right-hand side
            rhs = np.zeros(n)
            for i in range(n):
                if i == 0:
                    rhs[i] = b_R[i] * V[i] + c_R[i] * V[i+1] - a_R[i] * V_left
                elif i == n-1:
                    rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] - c_R[i] * V_right
                else:
                    rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] + c_R[i] * V[i+1]

            # Thomas algorithm (tridiagonal solve)
            c_prime = np.zeros(n)
            d_prime = np.zeros(n)

            c_prime[0] = c_L[0] / b_L[0]
            d_prime[0] = rhs[0] / b_L[0]

            for i in range(1, n):
                denom = b_L[i] - a_L[i] * c_prime[i-1]
                c_prime[i] = c_L[i] / denom if i < n-1 else 0.0
                d_prime[i] = (rhs[i] - a_L[i] * d_prime[i-1]) / denom

            V[n-1] = d_prime[n-1]
            for i in range(n-2, -1, -1):
                V[i] = d_prime[i] - c_prime[i] * V[i+1]

        # Interpolate to S0
        S_interior = self.S_grid[1:-1]
        price = self._interpolate_cubic(S_interior, V, S0)

        return price, V

    def solve_on_grid(self, S_output_grid: np.ndarray) -> np.ndarray:
        """
        Solve PDE and interpolate to output grid

        This is the key method for unified grid bumping.

        Args:
            S_output_grid: Output S grid (numpy array)

        Returns:
            V_output: Option values on output grid
        """
        # Solve PDE on internal grid
        _, V_interior = self._solve_pde_numerical(self.S0, self.sigma)

        # Interpolate to output grid
        V_output = self._interpolate_to_grid(V_interior, S_output_grid)

        return V_output

    def _interpolate_cubic(self, S_points: np.ndarray, V_values: np.ndarray,
                           S_target: float) -> float:
        """
        Cubic spline interpolation to a single target point

        IMPORTANT: Uses the SAME custom implementation as Edge-Pushing
        to ensure identical results.

        Args:
            S_points: Grid points
            V_values: Function values
            S_target: Target point

        Returns:
            Interpolated value
        """
        # Use custom natural cubic spline (matches Edge-Pushing)
        M_vals = self._compute_spline_second_derivatives(V_values, S_points)

        # Find interval containing S_target
        idx = np.searchsorted(S_points, S_target)
        if idx == 0:
            idx = 1
        elif idx >= len(S_points):
            idx = len(S_points) - 1

        i = idx - 1
        S_i = S_points[i]
        S_i1 = S_points[i + 1]
        V_i = V_values[i]
        V_i1 = V_values[i + 1]
        M_i = M_vals[i]
        M_i1 = M_vals[i + 1]

        h = S_i1 - S_i

        # Cubic spline evaluation
        A = (S_i1 - S_target) / h
        B = (S_target - S_i) / h

        price = (A * V_i + B * V_i1 +
                ((A**3 - A) * h**2 / 6.0) * M_i +
                ((B**3 - B) * h**2 / 6.0) * M_i1)

        return float(price)

    def _compute_spline_second_derivatives(self, V: np.ndarray,
                                          S_grid: np.ndarray) -> np.ndarray:
        """
        Compute natural cubic spline second derivatives M_i.

        IMPORTANT: This is the SAME implementation as Edge-Pushing
        (without ADVar, just using floats).

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

        A_diag = np.zeros(n_interior)
        A_lower = np.zeros(n_interior - 1)
        A_upper = np.zeros(n_interior - 1)
        rhs = np.zeros(n_interior)

        for i in range(1, n - 1):
            h_im1 = h[i - 1]
            h_i = h[i]

            lambda_i = h_im1 / (h_im1 + h_i)
            mu_i = h_i / (h_im1 + h_i)

            # Diagonal is always 2
            A_diag[i - 1] = 2.0

            # Off-diagonals
            if i > 1:
                A_lower[i - 2] = lambda_i
            if i < n - 2:
                A_upper[i - 1] = mu_i

            # RHS
            d_i = (6.0 / (h_im1 + h_i)) * (
                (V[i + 1] - V[i]) / h_i - (V[i] - V[i - 1]) / h_im1
            )
            rhs[i - 1] = d_i

        if n_interior == 1:
            M_interior = np.array([rhs[0] / A_diag[0]])
        else:
            # Solve tridiagonal system using Thomas algorithm
            M_interior = self._tridiag_solve(A_lower, A_diag, A_upper, rhs)

        # Add natural boundary conditions
        M_vals = np.concatenate([[0.0], M_interior, [0.0]])

        return M_vals

    def _tridiag_solve(self, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                      d: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system Ax = d where A has:
        - main diagonal b
        - lower diagonal a (offset -1)
        - upper diagonal c (offset +1)

        Uses Thomas algorithm.
        """
        n = len(b)
        c_prime = np.zeros(n - 1)
        d_prime = np.zeros(n)
        x = np.zeros(n)

        # Forward sweep
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n - 1):
            denom = b[i] - a[i - 1] * c_prime[i - 1]
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom

        # Last row
        denom = b[n - 1] - a[n - 2] * c_prime[n - 2]
        d_prime[n - 1] = (d[n - 1] - a[n - 2] * d_prime[n - 2]) / denom

        # Back substitution
        x[n - 1] = d_prime[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]

        return x

    def _interpolate_to_grid(self, V_interior: np.ndarray,
                             S_output_grid: np.ndarray) -> np.ndarray:
        """
        Interpolate interior solution to output grid

        Args:
            V_interior: Solution on interior grid (M-2,)
            S_output_grid: Output S grid (N_out,)

        Returns:
            V_output: Values on output grid (N_out,)
        """
        S_interior = self.S_grid[1:-1]

        # Build cubic spline
        cs = CubicSpline(S_interior, V_interior, bc_type='natural')

        # Interpolate
        V_output = cs(S_output_grid)

        # Handle extrapolation
        mask_low = S_output_grid < S_interior[0]
        mask_high = S_output_grid > S_interior[-1]

        if np.any(mask_low):
            # S < S_min: V ≈ 0
            V_output[mask_low] = 0.0

        if np.any(mask_high):
            # S > S_max: V ≈ S - K*exp(-r*T)
            V_output[mask_high] = (S_output_grid[mask_high] -
                                    self.K * np.exp(-self.r * self.T))

        return V_output
