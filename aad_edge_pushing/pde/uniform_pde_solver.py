"""
Uniform Grid PDE Solver for Black-Scholes Equation
Pure numerical solver with uniform spatial grid (not log-space)

Key difference from SimplePDESolver:
- Uses uniform grid in S-space (not log-space)
- PDE coefficients vary at each grid point (S-dependent)
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import CubicSpline

from .pde_config import PDEConfig


class UniformPDESolver:
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 M: int = 151, N_base: int = 50, n_rannacher: int = 4,
                 use_adaptive_Smax: bool = True, sigma_margin: float = 0.1,
                 S_max_override: Optional[float] = None):
        """
        Initialize PDE solver with uniform spatial grid

        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            M: Number of spatial grid points
            N_base: Number of time steps
            n_rannacher: Number of Rannacher smoothing steps (0 to disable)
            use_adaptive_Smax: If True, S_max adapts to sigma; if False, S_max = 5K
            sigma_margin: Safety margin for bumping (used when use_adaptive_Smax=True)
            S_max_override: If provided, use this S_max instead of computing it
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N_base = N_base
        self.n_rannacher = n_rannacher

        # Adaptive S_max with margin for bumping consistency
        if S_max_override is not None:
            S_max = S_max_override
        elif use_adaptive_Smax:
            S_max = PDEConfig.compute_unified_Smax(K, T, sigma, r, sigma_margin)
        else:
            S_max = 5.0 * K

        self.S_max = S_max

        # Create UNIFORM grid in S-space
        S_min = 0.0  # Can start from 0 for uniform grid
        self.S_grid, self.dS = PDEConfig.create_uniform_grid(S_min, S_max, M)

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
        Solve PDE with uniform spatial grid

        Black-Scholes PDE in S-space:
            ∂V/∂t + 0.5σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0

        Finite difference discretization:
            ∂²V/∂S² ≈ (V[i+1] - 2V[i] + V[i-1]) / dS²
            ∂V/∂S ≈ (V[i+1] - V[i-1]) / (2dS)

        Args:
            S0: Stock price to evaluate
            sigma: Volatility (can differ from self.sigma for bumping)

        Returns:
            price: Option value at S0
            V_interior: Solution on interior grid points
        """
        N = self.N_base
        dt = self.T / N
        t_grid = np.linspace(0, self.T, N + 1)

        n = self.M - 2  # Interior points
        dS = self.dS

        # Terminal condition
        V_terminal = self._terminal_condition()
        V = V_terminal[1:-1].copy()

        # Interior S values (for computing coefficients)
        S_interior = self.S_grid[1:-1]

        # Time stepping with Rannacher smoothing
        for n_step in range(N):
            phi = PDEConfig.get_rannacher_phi(n_step, self.n_rannacher)

            t_current = t_grid[n_step+1]
            V_left = self._boundary_condition_left(t_current)
            V_right = self._boundary_condition_right(t_current)

            # Build coefficient matrices
            # Coefficients are S-dependent for uniform grid!
            a_L = np.zeros(n)
            b_L = np.zeros(n)
            c_L = np.zeros(n)
            a_R = np.zeros(n)
            b_R = np.zeros(n)
            c_R = np.zeros(n)

            for i in range(n):
                S_i = S_interior[i]

                # PDE coefficients at grid point i
                alpha_i = 0.5 * sigma**2 * S_i**2 / (dS**2)  # Diffusion
                beta_i = self.r * S_i / (2.0 * dS)  # Drift
                gamma_i = -self.r  # Discount

                l_i = alpha_i - beta_i  # Lower diagonal
                c_i = -2.0 * alpha_i + gamma_i  # Main diagonal
                u_i = alpha_i + beta_i  # Upper diagonal

                # Left-hand side (implicit part)
                a_L[i] = -phi * dt * l_i if i > 0 else 0.0
                b_L[i] = 1.0 - phi * dt * c_i
                c_L[i] = -phi * dt * u_i if i < n-1 else 0.0

                # Right-hand side (explicit part)
                a_R[i] = (1.0 - phi) * dt * l_i if i > 0 else 0.0
                b_R[i] = 1.0 + (1.0 - phi) * dt * c_i
                c_R[i] = (1.0 - phi) * dt * u_i if i < n-1 else 0.0

            # Right-hand side vector
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
        price = self._interpolate_cubic(S_interior, V, S0)

        return price, V

    def _interpolate_cubic(self, S_points: np.ndarray, V_values: np.ndarray,
                           S_target: float) -> float:
        """
        Cubic spline interpolation

        Args:
            S_points: Grid points
            V_values: Function values
            S_target: Target point

        Returns:
            Interpolated value
        """
        cs = CubicSpline(S_points, V_values, bc_type='natural')
        return float(cs(S_target))

    def solve_on_grid(self, S_output_grid: np.ndarray) -> np.ndarray:
        """
        Solve PDE and interpolate to output grid

        Args:
            S_output_grid: Output S grid (numpy array)

        Returns:
            V_output: Option values on output grid
        """
        _, V_interior = self._solve_pde_numerical(self.S0, self.sigma)
        V_output = self._interpolate_to_grid(V_interior, S_output_grid)
        return V_output

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
        cs = CubicSpline(S_interior, V_interior, bc_type='natural')
        V_output = cs(S_output_grid)

        # Handle extrapolation
        mask_low = S_output_grid < S_interior[0]
        mask_high = S_output_grid > S_interior[-1]

        if np.any(mask_low):
            V_output[mask_low] = 0.0

        if np.any(mask_high):
            V_output[mask_high] = (S_output_grid[mask_high] -
                                    self.K * np.exp(-self.r * self.T))

        return V_output
