"""
B-spline volatility model for local volatility.

The B-spline model provides a flexible parametric form for the volatility surface:
    σ(S) = Σᵢ wᵢ Bᵢ(S)

where:
- Bᵢ(S): B-spline basis functions of degree n
- wᵢ: B-spline coefficients (weights)
- Knot vector Γ = {γ₀, γ₁, ..., γₖ₋₁}

Key Features:
1. LOCAL SUPPORT: Each basis function Bᵢ(x) is non-zero only on [γᵢ₋ₙ, γᵢ]
2. SPARSE HESSIAN: Parameter Hessian has banded structure with bandwidth ≈ 2n+1
3. SMOOTHNESS: C^(n-1) continuous (e.g., cubic B-splines are C² continuous)

This sparse Hessian structure makes the model ideal for AAD edge-pushing methods,
as the Hessian computation complexity scales as O(N × bandwidth) instead of O(N²).

References:
- De Boor (1978): "A Practical Guide to Splines"
- Andersen & Piterbarg (2010): "Interest Rate Modeling" (Vol 1, Ch 3)
- Bayer & Laurence (2018): "Quasi-explicit calibration of Gatheral's SVI model"
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.interpolate import BSpline, splrep, splev


class BSplineModel:
    """
    B-spline volatility surface model with sparse Hessian properties.

    For a 1D volatility model σ(S), the B-spline representation is:
        σ(S) = Σᵢ₌₀^(k+n) wᵢ Bᵢ^Γ_n(S)

    where:
    - k: number of interior knots
    - n: degree of B-spline (3 = cubic, 2 = quadratic)
    - wᵢ: coefficients (the parameters for AAD/Hessian computation)
    - Bᵢ^Γ_n(S): B-spline basis functions

    The parameter Hessian H[i,j] = ∂²price/∂wᵢ∂wⱼ is SPARSE with:
    - H[i,j] = 0 when |i-j| > n (due to compact support)
    - Bandwidth ≈ 2n+1 (typically 7 for cubic splines)
    - Sparsity ratio ≈ (2n+1)/(k+n+1), which is high for k >> n
    """

    def __init__(self,
                 knots: np.ndarray,
                 coefficients: np.ndarray,
                 degree: int = 3,
                 S_min: float = None,
                 S_max: float = None):
        """
        Initialize B-spline volatility model.

        Args:
            knots: Interior knot vector (k points), should span [S_min, S_max]
            coefficients: B-spline coefficients (weights) wᵢ, length = k + degree + 1
            degree: Degree of B-spline (default=3 for cubic, C² continuity)
            S_min: Minimum spot price (for extrapolation boundaries)
            S_max: Maximum spot price (for extrapolation boundaries)

        Example:
            # Create a cubic B-spline with 5 knots on [50, 150]
            knots = np.linspace(50, 150, 5)
            coefficients = np.array([0.2, 0.22, 0.25, 0.23, 0.21, 0.20, 0.19, 0.18])
            model = BSplineModel(knots, coefficients, degree=3)
        """
        self.degree = degree
        self.interior_knots = np.asarray(knots, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

        # Number of interior knots
        self.k = len(self.interior_knots)

        # Expected number of coefficients: k + degree + 1
        # (For open uniform knots with multiplicity degree+1 at boundaries)
        expected_n_coeffs = self.k + self.degree + 1

        if len(self.coefficients) != expected_n_coeffs:
            raise ValueError(
                f"Number of coefficients ({len(self.coefficients)}) must equal "
                f"k + degree + 1 = {self.k} + {self.degree} + 1 = {expected_n_coeffs}"
            )

        # Boundaries for extrapolation
        if S_min is None:
            S_min = self.interior_knots[0]
        if S_max is None:
            S_max = self.interior_knots[-1]

        self.S_min = S_min
        self.S_max = S_max

        # Build full knot vector with boundary conditions
        # For natural/clamped splines: repeat boundary knots (degree+1) times
        self.knot_vector = self._build_knot_vector()

        # Create scipy BSpline object for evaluation
        self._bspline = BSpline(self.knot_vector, self.coefficients, self.degree,
                                extrapolate=True)

        # Validate parameters
        self._validate_parameters()

    def _build_knot_vector(self) -> np.ndarray:
        """
        Build full knot vector with boundary conditions.

        For clamped B-splines (most common for interpolation):
        - Repeat first knot (degree+1) times
        - Interior knots
        - Repeat last knot (degree+1) times

        Returns:
            Full knot vector for BSpline construction
        """
        k_first = self.interior_knots[0]
        k_last = self.interior_knots[-1]

        # Clamped boundary conditions
        left_boundary = np.repeat(k_first, self.degree + 1)
        right_boundary = np.repeat(k_last, self.degree + 1)

        # Full knot vector
        knot_vector = np.concatenate([left_boundary,
                                     self.interior_knots[1:-1],  # Exclude already-used boundaries
                                     right_boundary])

        return knot_vector

    def _validate_parameters(self):
        """Validate B-spline parameters and constraints."""
        # Check degree
        if self.degree < 1:
            raise ValueError(f"Degree must be >= 1, got {self.degree}")

        # Check coefficients are positive (volatilities must be positive)
        if np.any(self.coefficients <= 0):
            negative_idx = np.where(self.coefficients <= 0)[0]
            raise ValueError(
                f"All coefficients must be positive (represent volatilities). "
                f"Found non-positive values at indices {negative_idx}: "
                f"{self.coefficients[negative_idx]}"
            )

        # Check for reasonable volatility values (1% to 200%)
        if np.any(self.coefficients < 0.01) or np.any(self.coefficients > 2.0):
            import warnings
            warnings.warn(
                f"Coefficients outside typical range [0.01, 2.0]. "
                f"Range: [{self.coefficients.min():.4f}, {self.coefficients.max():.4f}]"
            )

        # Check knot vector is sorted
        if not np.all(np.diff(self.interior_knots) >= 0):
            raise ValueError("Knot vector must be non-decreasing")

    def evaluate(self, S: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Evaluate B-spline volatility at given spot prices.

        Args:
            S: Spot prices (can be scalar or array)
            t: Time (unused in 1D spatial model, kept for interface compatibility)

        Returns:
            Volatilities σ(S) at given spot prices
        """
        S = np.atleast_1d(S)

        # Evaluate B-spline
        sigma = self._bspline(S)

        # Apply floor for numerical stability
        sigma = np.maximum(sigma, 0.01)  # 1% minimum volatility

        return sigma

    def to_pde_grid(self, S_grid: np.ndarray, T_grid: np.ndarray,
                   r: float, S_ref: float = None) -> np.ndarray:
        """
        Convert B-spline model to local volatility grid for PDE solver.

        For 1D spatial B-spline σ(S), the volatility is time-independent.
        We return the same spatial profile for all time steps.

        Args:
            S_grid: Spatial grid [S_0, S_1, ..., S_M]
            T_grid: Time grid [T_0, T_1, ..., T_N]
            r: Risk-free rate (unused in 1D spatial model)
            S_ref: Reference spot price (unused, kept for interface compatibility)

        Returns:
            Local volatility grid σ[i, n] of shape (M+1, N+1)
            For 1D spatial model: σ[i, n] = σ(S_i) for all n
        """
        M = len(S_grid) - 1
        N = len(T_grid) - 1

        # Evaluate B-spline at all spatial grid points
        sigma_S = self.evaluate(S_grid)

        # Replicate across time dimension
        sigma_grid = np.tile(sigma_S.reshape(-1, 1), (1, N + 1))

        return sigma_grid

    def get_support_indices(self, S: float) -> List[int]:
        """
        Get indices of basis functions with non-zero support at S.

        Due to compact support, only (degree+1) basis functions are non-zero at any S.
        This is the key property that leads to sparse Hessians.

        Args:
            S: Spot price

        Returns:
            List of coefficient indices i such that Bᵢ(S) ≠ 0
        """
        # Find which knot interval S belongs to
        # For clamped B-splines, this determines the active basis functions

        # Handle boundary cases
        if S <= self.knot_vector[self.degree]:
            # Left boundary
            return list(range(self.degree + 1))
        elif S >= self.knot_vector[-(self.degree + 1)]:
            # Right boundary
            n_coeffs = len(self.coefficients)
            return list(range(n_coeffs - self.degree - 1, n_coeffs))

        # Find knot interval
        interval_idx = np.searchsorted(self.knot_vector, S, side='right') - 1

        # Active basis functions span [interval_idx - degree, interval_idx]
        start_idx = max(0, interval_idx - self.degree)
        end_idx = min(len(self.coefficients), interval_idx + 1)

        return list(range(start_idx, end_idx))

    def get_hessian_sparsity_pattern(self) -> np.ndarray:
        """
        Get the sparsity pattern of the parameter Hessian.

        For B-spline models, H[i,j] = ∂²price/∂wᵢ∂wⱼ is sparse because:
        - Bᵢ(S) and Bⱼ(S) have disjoint support when |i-j| > degree
        - Therefore H[i,j] = 0 when |i-j| > degree

        Returns:
            Boolean array of shape (n_params, n_params) indicating non-zero entries
        """
        n_params = len(self.coefficients)
        sparsity = np.zeros((n_params, n_params), dtype=bool)

        # Fill banded structure
        for i in range(n_params):
            for j in range(max(0, i - self.degree), min(n_params, i + self.degree + 1)):
                sparsity[i, j] = True

        return sparsity

    def get_hessian_bandwidth(self) -> int:
        """
        Get the bandwidth of the Hessian matrix.

        Returns:
            Bandwidth (2 * degree + 1 for symmetric banded matrix)
        """
        return 2 * self.degree + 1

    def get_n_params(self) -> int:
        """Get number of parameters (coefficients)."""
        return len(self.coefficients)

    def update_coefficients(self, new_coefficients: np.ndarray):
        """
        Update B-spline coefficients (for calibration/optimization).

        Args:
            new_coefficients: New coefficient values
        """
        if len(new_coefficients) != len(self.coefficients):
            raise ValueError(
                f"New coefficients must have same length as original "
                f"({len(self.coefficients)}), got {len(new_coefficients)}"
            )

        self.coefficients = np.asarray(new_coefficients, dtype=np.float64)

        # Rebuild BSpline object
        self._bspline = BSpline(self.knot_vector, self.coefficients, self.degree,
                                extrapolate=True)

        # Validate
        self._validate_parameters()


def create_sample_bspline(S_min: float = 50.0, S_max: float = 150.0,
                         n_knots: int = 10, degree: int = 3,
                         base_vol: float = 0.20,
                         vol_variation: float = 0.05) -> BSplineModel:
    """
    Create a sample B-spline model with realistic parameters.

    Args:
        S_min: Minimum spot price
        S_max: Maximum spot price
        n_knots: Number of interior knots
        degree: B-spline degree (3 = cubic)
        base_vol: Base volatility level (e.g., 0.20 = 20%)
        vol_variation: Random variation around base (e.g., 0.05 = ±5%)

    Returns:
        BSplineModel with smooth volatility surface
    """
    # Create uniform knot vector
    knots = np.linspace(S_min, S_max, n_knots)

    # Generate smooth coefficients with slight smile
    n_coeffs = n_knots + degree + 1

    # Create smile pattern: higher vol at wings, lower at center
    x = np.linspace(-1, 1, n_coeffs)
    smile = base_vol + vol_variation * (x**2 - 0.5)  # Parabolic smile

    # Add small random perturbations for realism
    np.random.seed(42)  # Reproducible
    perturbations = np.random.normal(0, vol_variation * 0.1, n_coeffs)
    coefficients = smile + perturbations

    # Ensure all coefficients are positive and reasonable
    coefficients = np.clip(coefficients, 0.05, 0.50)

    return BSplineModel(knots, coefficients, degree, S_min, S_max)


def create_flat_bspline(n_knots: int = 10, degree: int = 3,
                       volatility: float = 0.20,
                       S_min: float = 50.0, S_max: float = 150.0) -> BSplineModel:
    """
    Create a flat B-spline model (constant volatility).

    Useful for testing: should match constant volatility BS model.

    Args:
        n_knots: Number of interior knots
        degree: B-spline degree
        volatility: Constant volatility value
        S_min: Minimum spot price
        S_max: Maximum spot price

    Returns:
        BSplineModel with constant volatility
    """
    knots = np.linspace(S_min, S_max, n_knots)
    n_coeffs = n_knots + degree + 1
    coefficients = np.full(n_coeffs, volatility)

    return BSplineModel(knots, coefficients, degree, S_min, S_max)


if __name__ == "__main__":
    # Test B-spline model
    print("="*70)
    print("B-Spline Volatility Model Test")
    print("="*70)

    # Create sample B-spline
    print("\n1. Creating sample B-spline model...")
    bspline = create_sample_bspline(S_min=50, S_max=150, n_knots=10, degree=3)

    print(f"   - Degree: {bspline.degree}")
    print(f"   - Number of interior knots: {bspline.k}")
    print(f"   - Number of coefficients: {len(bspline.coefficients)}")
    print(f"   - Coefficients: {bspline.coefficients}")
    print(f"   - Hessian bandwidth: {bspline.get_hessian_bandwidth()}")

    # Test evaluation
    print("\n2. Evaluating volatility at test points...")
    S_test = np.array([60, 80, 100, 120, 140])
    sigma_test = bspline.evaluate(S_test)

    print(f"{'Spot Price':<15} {'Volatility':<15}")
    print("-"*30)
    for S, sig in zip(S_test, sigma_test):
        print(f"{S:<15.1f} {sig:<15.4f}")

    # Test PDE grid conversion
    print("\n3. Converting to PDE grid...")
    S_grid = np.linspace(50, 150, 21)
    T_grid = np.linspace(0, 1.0, 6)
    r = 0.05

    sigma_grid = bspline.to_pde_grid(S_grid, T_grid, r)
    print(f"   Grid shape: {sigma_grid.shape}")
    print(f"   Volatility range: [{sigma_grid.min():.4f}, {sigma_grid.max():.4f}]")

    # Test sparsity pattern
    print("\n4. Analyzing Hessian sparsity...")
    sparsity = bspline.get_hessian_sparsity_pattern()
    n_params = bspline.get_n_params()
    n_nonzero = np.sum(sparsity)
    n_total = n_params * n_params
    sparsity_ratio = n_nonzero / n_total

    print(f"   - Number of parameters: {n_params}")
    print(f"   - Total Hessian entries: {n_total}")
    print(f"   - Non-zero entries: {n_nonzero}")
    print(f"   - Sparsity ratio: {sparsity_ratio:.2%}")
    print(f"   - Zero entries: {n_total - n_nonzero} ({(1-sparsity_ratio):.2%})")

    # Test flat B-spline
    print("\n5. Testing flat B-spline (constant volatility)...")
    flat = create_flat_bspline(n_knots=10, degree=3, volatility=0.25)
    sigma_flat = flat.evaluate(S_test)

    print(f"   Expected: 0.2500 (constant)")
    print(f"   Actual: {sigma_flat}")
    print(f"   Max deviation: {np.max(np.abs(sigma_flat - 0.25)):.2e}")

    # Test support indices
    print("\n6. Testing compact support property...")
    S_point = 100.0
    active_indices = bspline.get_support_indices(S_point)
    print(f"   At S={S_point}, active basis functions: {active_indices}")
    print(f"   Number of active functions: {len(active_indices)} (expected: {bspline.degree + 1})")

    print("\n" + "="*70)
    print("✓ B-spline model test completed successfully!")
    print("="*70)
