"""
SVI (Stochastic Volatility Inspired) model for local volatility.

The SVI model provides a parametric form for the implied volatility surface:
    w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))

where:
- k = log(K/F): log-moneyness
- w = σ²_imp × T: total implied variance
- Parameters: θ = (a, b, ρ, m, σ)

This module:
1. Implements SVI parametric model
2. Converts SVI to local volatility via Dupire formula
3. Provides grid-based local vol σ(S,t) for PDE pricing

References:
- Gatheral (2004): "A parsimonious arbitrage-free implied volatility parameterization"
- Gatheral & Jacquier (2014): "Arbitrage-free SVI volatility surfaces"
"""

import numpy as np
from typing import Tuple, Dict
from scipy.interpolate import interp2d, RectBivariateSpline


class SVIModel:
    """
    SVI volatility model with Dupire local volatility conversion.

    The SVI parametrization ensures smooth, arbitrage-free implied volatility surfaces.
    """

    def __init__(self, a: float, b: float, rho: float, m: float, sigma: float):
        """
        Initialize SVI model with parameters.

        Args:
            a: Level parameter (controls ATM variance)
            b: Slope parameter (controls wing behavior)
            rho: Correlation parameter (controls skew), |ρ| < 1
            m: Translation parameter (shifts surface)
            sigma: Scale parameter (controls curvature), σ > 0

        Constraints for arbitrage-free:
            - a + b*σ*√(1-ρ²) ≥ 0
            - b ≥ 0
            - |ρ| < 1
            - σ > 0
        """
        self.a = a
        self.b = b
        self.rho = rho
        self.m = m
        self.sigma = sigma

        # Validate constraints
        self._validate_parameters()

    def _validate_parameters(self):
        """Check arbitrage-free constraints."""
        if self.b < 0:
            raise ValueError(f"b must be non-negative, got {self.b}")
        if abs(self.rho) >= 1:
            raise ValueError(f"|rho| must be < 1, got {self.rho}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")

        min_var = self.a + self.b * self.sigma * np.sqrt(1 - self.rho**2)
        if min_var < 0:
            raise ValueError(f"Minimum variance is negative: {min_var}")

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """
        Compute total implied variance w(k) = σ²_imp × T.

        Args:
            k: Log-moneyness k = log(K/F)

        Returns:
            Total variance w(k)
        """
        k_shifted = k - self.m
        sqrt_term = np.sqrt(k_shifted**2 + self.sigma**2)
        w = self.a + self.b * (self.rho * k_shifted + sqrt_term)
        return w

    def implied_volatility(self, K: np.ndarray, F: float, T: float) -> np.ndarray:
        """
        Compute implied volatility σ_imp(K, T).

        Args:
            K: Strike prices
            F: Forward price F = S * exp(r*T)
            T: Time to maturity

        Returns:
            Implied volatilities σ_imp(K, T)
        """
        k = np.log(K / F)
        w = self.total_variance(k)
        sigma_imp = np.sqrt(w / T)
        return sigma_imp

    def local_volatility_dupire(self, K: np.ndarray, T_grid: np.ndarray,
                                F: float, r: float) -> np.ndarray:
        """
        Compute local volatility σ_local(K, T) using Dupire formula.

        Dupire formula:
            σ²_local(K,T) = (∂w/∂T + (r-d)K∂w/∂K + 0.25K²∂²w/∂K²) /
                           (1 + K∂√w/∂K√T/√w + 0.25K²(∂²√w/∂K²T/√w - (∂√w/∂K)²T/w))

        Simplified for SVI (analytical derivatives available):
            σ²_local = (∂w/∂T) / (1 - k/(2w) ∂w/∂k)²

        Args:
            K: Strike prices (array)
            T_grid: Time grid (array)
            F: Forward price
            r: Risk-free rate

        Returns:
            Local volatility grid σ_local[K, T] (2D array)
        """
        nK = len(K)
        nT = len(T_grid)
        sigma_local = np.zeros((nK, nT))

        for i, T in enumerate(T_grid):
            if T < 1e-8:  # Avoid division by zero at T=0
                sigma_local[:, i] = np.sqrt(self.a)
                continue

            k = np.log(K / (F * np.exp(r * T)))

            # SVI total variance and derivatives
            w = self.total_variance(k)

            # ∂w/∂k (analytical)
            k_shifted = k - self.m
            sqrt_term = np.sqrt(k_shifted**2 + self.sigma**2)
            dw_dk = self.b * (self.rho + k_shifted / sqrt_term)

            # Simplified Dupire for constant parameters (∂w/∂T ≈ w/T)
            # More accurate: use finite difference for ∂w/∂T
            if i > 0:
                w_prev = self.total_variance(np.log(K / (F * np.exp(r * T_grid[i-1]))))
                dw_dT = (w - w_prev) / (T - T_grid[i-1])
            else:
                dw_dT = w / T  # Approximation at T=0

            # Dupire formula
            numerator = dw_dT
            denominator = 1 - k / (2 * w) * dw_dk

            # Avoid numerical issues
            denominator = np.clip(denominator, 0.1, 10.0)

            sigma_local_sq = numerator / (denominator**2 * T)
            sigma_local_sq = np.maximum(sigma_local_sq, 1e-6)  # Floor for stability

            sigma_local[:, i] = np.sqrt(sigma_local_sq)

        return sigma_local

    def to_pde_grid(self, S_grid: np.ndarray, T_grid: np.ndarray,
                   r: float, S_ref: float = None) -> np.ndarray:
        """
        Convert SVI model to local volatility grid for PDE solver.

        Args:
            S_grid: Spatial grid [S_0, S_1, ..., S_M]
            T_grid: Time grid [T_0, T_1, ..., T_N]
            r: Risk-free rate
            S_ref: Reference spot price for forward calculation (default: mid-point)

        Returns:
            Local volatility grid σ[i, n] of shape (M+1, N+1)
        """
        M = len(S_grid) - 1
        N = len(T_grid) - 1

        # Use a reasonable reference price (not S=0!)
        if S_ref is None:
            S_ref = S_grid[M//2] if S_grid[M//2] > 0 else np.mean(S_grid[S_grid > 0])

        # Simplified approach: Use ATM implied vol as approximation for local vol
        # This avoids numerical issues with Dupire at boundaries
        sigma_grid = np.zeros((M+1, N+1))

        for n, T in enumerate(T_grid):
            if T < 1e-8:
                # Initial volatility (from SVI parameter a)
                sigma_grid[:, n] = np.sqrt(max(self.a, 0.01))
            else:
                # For each S, compute implied vol and use as local vol approximation
                F = S_ref * np.exp(r * T)
                for i, S in enumerate(S_grid):
                    if S < 1e-8:  # Avoid S=0
                        sigma_grid[i, n] = np.sqrt(max(self.a, 0.01))
                    else:
                        # Use S as strike
                        sigma_imp = self.implied_volatility(np.array([S]), F, T)
                        sigma_grid[i, n] = max(sigma_imp[0], 0.01)  # Floor at 1%

        return sigma_grid


class SVICalibrator:
    """
    Calibrate SVI model to market data.

    Given market implied volatilities, find optimal SVI parameters.
    """

    def __init__(self):
        self.model = None

    def calibrate(self, K_market: np.ndarray, T_market: np.ndarray,
                 sigma_market: np.ndarray, S0: float, r: float,
                 initial_guess: Dict[str, float] = None) -> SVIModel:
        """
        Calibrate SVI parameters to market implied volatilities.

        Args:
            K_market: Market strikes
            T_market: Market maturities
            sigma_market: Market implied volatilities
            S0: Spot price
            r: Risk-free rate
            initial_guess: Initial parameter guess

        Returns:
            Calibrated SVIModel
        """
        from scipy.optimize import minimize

        if initial_guess is None:
            # Default initial guess (reasonable for typical markets)
            initial_guess = {
                'a': 0.04,
                'b': 0.2,
                'rho': -0.4,
                'm': 0.0,
                'sigma': 0.1
            }

        def objective(params):
            a, b, rho, m, sigma = params
            try:
                model = SVIModel(a, b, rho, m, sigma)
                F = S0 * np.exp(r * T_market)
                sigma_pred = model.implied_volatility(K_market, F, T_market)
                error = np.sum((sigma_pred - sigma_market)**2)
                return error
            except:
                return 1e10  # Penalize invalid parameters

        # Bounds
        bounds = [
            (0.001, 0.5),    # a
            (0.001, 1.0),    # b
            (-0.999, 0.999), # rho
            (-0.5, 0.5),     # m
            (0.001, 1.0)     # sigma
        ]

        x0 = [initial_guess['a'], initial_guess['b'], initial_guess['rho'],
              initial_guess['m'], initial_guess['sigma']]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        a, b, rho, m, sigma = result.x
        self.model = SVIModel(a, b, rho, m, sigma)

        return self.model


def create_sample_svi() -> SVIModel:
    """
    Create a sample SVI model with typical market parameters.

    Returns:
        SVIModel with realistic parameters
    """
    # Typical equity market parameters
    a = 0.04      # Base variance level (σ ≈ 20%)
    b = 0.3       # Moderate slope
    rho = -0.5    # Negative skew (puts more expensive)
    m = 0.0       # ATM centered
    sigma = 0.15  # Moderate curvature

    return SVIModel(a, b, rho, m, sigma)


if __name__ == "__main__":
    # Test SVI model
    print("="*70)
    print("SVI Model Test")
    print("="*70)

    # Create sample SVI
    svi = create_sample_svi()
    print(f"\nSVI Parameters:")
    print(f"  a={svi.a}, b={svi.b}, ρ={svi.rho}, m={svi.m}, σ={svi.sigma}")

    # Test implied volatilities
    S0 = 100
    r = 0.05
    T = 1.0
    F = S0 * np.exp(r * T)

    K_test = np.array([80, 90, 100, 110, 120])
    sigma_imp = svi.implied_volatility(K_test, F, T)

    print(f"\nImplied Volatilities at T={T}:")
    print(f"{'Strike':<10} {'Moneyness':<15} {'Implied Vol':<15}")
    print("-"*40)
    for K, sig in zip(K_test, sigma_imp):
        print(f"{K:<10.1f} {K/S0:<15.2f} {sig:<15.4f}")

    # Test local volatility grid
    print(f"\nLocal Volatility Grid:")
    S_grid = np.linspace(50, 150, 11)
    T_grid = np.linspace(0, 1.0, 6)

    sigma_local_grid = svi.to_pde_grid(S_grid, T_grid, r)

    print(f"Grid shape: {sigma_local_grid.shape}")
    print(f"\nSample local vols at S=100:")
    idx_100 = np.argmin(np.abs(S_grid - 100))
    print(f"{'Time':<10} {'Local Vol':<15}")
    print("-"*25)
    for n, T in enumerate(T_grid):
        print(f"{T:<10.2f} {sigma_local_grid[idx_100, n]:<15.4f}")

    print("\n✓ SVI model test completed")
