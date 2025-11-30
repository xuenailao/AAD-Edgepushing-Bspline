"""
PDE Configuration Utilities

Shared configuration and helper functions for PDE solvers.
Provides adaptive S_max calculation and grid setup.
"""

import numpy as np
from typing import Tuple


class PDEConfig:
    """Shared configuration for PDE solvers"""

    @staticmethod
    def compute_adaptive_Smax(K: float, T: float, sigma: float, r: float,
                               n_std: float = 6.0, min_multiplier: float = 5.0) -> float:
        """
        Compute volatility-dependent domain boundary

        Theory:
            Under GBM, log(S_T) ~ N(μ, σ²T) where:
            μ = log(S0) + (r - 0.5σ²)T

            For a K-strike option, we use K as reference:
            log(S_max) = log(K) + (r - 0.5σ²)T + n_std × σ√T

        Args:
            K: Strike price
            T: Time to maturity
            sigma: Volatility
            r: Risk-free rate
            n_std: Number of standard deviations (default 6 for ~99.9% coverage)
            min_multiplier: Minimum S_max as multiple of K

        Returns:
            S_max: Adaptive domain boundary

        Example:
            σ=0.2, T=1.0, K=100: S_max ≈ 400  (4K)
            σ=0.5, T=1.0, K=100: S_max ≈ 2700 (27K)
        """
        # Mean of log(S_T) distribution
        mu_log = np.log(K) + (r - 0.5 * sigma**2) * T

        # Standard deviation
        std_log = sigma * np.sqrt(T)

        # n_std confidence interval upper bound
        log_Smax = mu_log + n_std * std_log
        S_max = np.exp(log_Smax)

        # Safety: ensure minimum size
        S_max_min = min_multiplier * K
        return max(S_max, S_max_min)

    @staticmethod
    def compute_unified_Smax(K: float, T: float, sigma_center: float, r: float,
                              sigma_margin: float = 0.1, n_std: float = 6.0) -> float:
        """
        Compute unified S_max for bumping with margin

        Used when computing Greeks via bumping to ensure all perturbed
        PDEs use the same grid.

        Args:
            sigma_center: Center volatility
            sigma_margin: Safety margin for σ perturbations

        Returns:
            S_max suitable for σ ∈ [sigma_center - margin, sigma_center + margin]

        Example:
            For σ=0.50 with margin=0.05:
            - V(σ=0.49), V(σ=0.50), V(σ=0.51) all use S_max(σ=0.55)
            - Ensures consistent grid across bumping
        """
        sigma_effective = sigma_center + sigma_margin
        return PDEConfig.compute_adaptive_Smax(K, T, sigma_effective, r, n_std)

    @staticmethod
    def create_log_space_grid(S_min: float, S_max: float, M: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Create uniform grid in log-space

        Args:
            S_min: Minimum stock price (e.g., 1e-3)
            S_max: Maximum stock price
            M: Number of grid points

        Returns:
            x_grid: Log-space grid (uniform)
            S_grid: Stock-space grid (non-uniform)
            dx: Grid spacing in log-space
        """
        x_min = np.log(S_min)
        x_max = np.log(S_max)

        x_grid = np.linspace(x_min, x_max, M)
        dx = x_grid[1] - x_grid[0]
        S_grid = np.exp(x_grid)

        return x_grid, S_grid, dx

    @staticmethod
    def create_uniform_grid(S_min: float, S_max: float, M: int) -> Tuple[np.ndarray, float]:
        """
        Create uniform grid in real space (not log-space)

        Args:
            S_min: Minimum stock price
            S_max: Maximum stock price
            M: Number of grid points

        Returns:
            S_grid: Stock-space grid (uniform)
            dS: Grid spacing in stock-space
        """
        S_grid = np.linspace(S_min, S_max, M)
        dS = S_grid[1] - S_grid[0]

        return S_grid, dS

    @staticmethod
    def get_rannacher_phi(n_step: int, n_rannacher: int) -> float:
        """
        Compute Rannacher smoothing parameter phi

        Rannacher smoothing uses fully implicit time stepping for the first
        few steps to smooth out discontinuities in the payoff, then switches
        to Crank-Nicolson for better accuracy.

        Args:
            n_step: Current time step index (0, 1, 2, ...)
            n_rannacher: Number of Rannacher smoothing steps

        Returns:
            phi: Time stepping parameter
                 phi = 1.0 (fully implicit) for first n_rannacher steps
                 phi = 0.5 (Crank-Nicolson) afterwards

        Example:
            n_rannacher = 4:
            - Steps 0,1,2,3: phi = 1.0 (fully implicit)
            - Steps 4,5,...: phi = 0.5 (Crank-Nicolson)
        """
        if n_step < n_rannacher:
            return 1.0  # Fully implicit
        else:
            return 0.5  # Crank-Nicolson
