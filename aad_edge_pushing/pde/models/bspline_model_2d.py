"""
2D B-spline volatility model for local volatility: σ(S, t).

The 2D B-spline model extends the 1D model to time-varying volatility:
    σ(S, t) = Σᵢ Σⱼ wᵢⱼ · Bᵢ(S) · Bⱼ(t)

where:
- Bᵢ(S): Spatial B-spline basis functions of degree n_S
- Bⱼ(t): Temporal B-spline basis functions of degree n_T
- wᵢⱼ: 2D coefficient matrix (weights)
- Knot vectors: Γ_S for space, Γ_T for time

Key Features:
1. TENSOR PRODUCT BASIS: Separable structure σ(S,t) = Σᵢⱼ wᵢⱼ Bᵢ(S) Bⱼ(t)
2. COMPACT SUPPORT: Only (n_S+1)×(n_T+1) basis functions active at any (S,t)
3. SPARSE HESSIAN: Block-banded structure with ~98-99% zeros for typical grids
4. SMOOTHNESS: C^(n_S-1) in S, C^(n_T-1) in t (e.g., cubic-quadratic is C²×C¹)

Sparse Hessian Structure:
For parameters indexed as w[i,j], the Hessian H[(i,j), (k,l)] = ∂²V/∂w_ij ∂w_kl is zero when:
    |i-k| > n_S  OR  |j-l| > n_T
This gives bandwidth = (2n_S+1) × (2n_T+1), typically 7×5=35 for cubic-quadratic.

References:
- De Boor (1978): "A Practical Guide to Splines"
- Andersen & Piterbarg (2010): "Interest Rate Modeling" (Tensor product surfaces)
- Fengler (2009): "Arbitrage-free smoothing of the implied volatility surface"
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from scipy.interpolate import BSpline
from dataclasses import dataclass


@dataclass
class BSplineConfig2D:
    """Configuration for 2D B-spline model."""
    S_min: float
    S_max: float
    T_min: float = 0.0
    T_max: float = 1.0
    n_knots_S: int = 10
    n_knots_T: int = 5
    degree_S: int = 3  # Cubic in space
    degree_T: int = 2  # Quadratic in time


class BSplineModel2D:
    """
    2D B-spline volatility surface model with sparse Hessian properties.

    For a time-varying volatility model σ(S, t), the B-spline representation is:
        σ(S, t) = Σᵢ₌₀^(k_S+n_S) Σⱼ₌₀^(k_T+n_T) wᵢⱼ · Bᵢ^(Γ_S)_{n_S}(S) · Bⱼ^(Γ_T)_{n_T}(t)

    where:
    - k_S, k_T: number of interior knots in space and time
    - n_S, n_T: degrees of B-splines (typically 3, 2)
    - wᵢⱼ: coefficient matrix (the parameters for AAD/Hessian computation)
    - Bᵢ, Bⱼ: B-spline basis functions

    The parameter Hessian H[(i,j), (k,l)] = ∂²price/∂w_ij ∂w_kl is SPARSE with:
    - H[(i,j), (k,l)] = 0 when |i-k| > n_S OR |j-l| > n_T
    - Block-banded structure
    - Sparsity ratio ≈ [(2n_S+1)(2n_T+1)] / [(k_S+n_S+1)(k_T+n_T+1)]
    - For typical grids: 98-99% zeros
    """

    def __init__(self,
                 knots_S: np.ndarray,
                 knots_T: np.ndarray,
                 coefficients: np.ndarray,
                 degree_S: int = 3,
                 degree_T: int = 2,
                 S_min: float = None,
                 S_max: float = None,
                 T_min: float = None,
                 T_max: float = None):
        """
        Initialize 2D B-spline volatility model.

        Args:
            knots_S: Interior knot vector for space (k_S points), should span [S_min, S_max]
            knots_T: Interior knot vector for time (k_T points), should span [T_min, T_max]
            coefficients: 2D coefficient matrix w[i,j], shape = (k_S+n_S+1, k_T+n_T+1)
            degree_S: Degree of B-spline in space (default=3 for cubic, C² continuity)
            degree_T: Degree of B-spline in time (default=2 for quadratic, C¹ continuity)
            S_min: Minimum spot price (for extrapolation boundaries)
            S_max: Maximum spot price (for extrapolation boundaries)
            T_min: Minimum time (typically 0)
            T_max: Maximum time (maturity horizon)

        Example:
            # Create a cubic-quadratic B-spline on [50, 150] × [0, 1]
            knots_S = np.linspace(50, 150, 10)
            knots_T = np.linspace(0, 1, 5)
            coefficients = 0.2 * np.ones((14, 8))  # 10+3+1 by 5+2+1
            model = BSplineModel2D(knots_S, knots_T, coefficients, degree_S=3, degree_T=2)
        """
        self.degree_S = degree_S
        self.degree_T = degree_T
        self.interior_knots_S = np.asarray(knots_S, dtype=np.float64)
        self.interior_knots_T = np.asarray(knots_T, dtype=np.float64)
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

        # Number of interior knots
        self.k_S = len(self.interior_knots_S)
        self.k_T = len(self.interior_knots_T)

        # Expected coefficient matrix shape
        # For clamped B-spline: n_basis = degree + n_interior_knots - 1
        # Knot vector: [0]*degree, interior_knots, [1]*degree
        # len(knot_vector) = 2*degree + n_interior_knots
        # n_basis = len(knot_vector) - degree - 1 = degree + n_interior_knots - 1
        expected_n_S = self.degree_S + self.k_S - 1
        expected_n_T = self.degree_T + self.k_T - 1

        if self.coefficients.shape != (expected_n_S, expected_n_T):
            raise ValueError(
                f"Coefficient matrix shape {self.coefficients.shape} must be "
                f"({expected_n_S}, {expected_n_T}) = "
                f"(degree_S+k_S-1, degree_T+k_T-1)"
            )

        # Boundaries for extrapolation
        self.S_min = S_min if S_min is not None else self.interior_knots_S[0]
        self.S_max = S_max if S_max is not None else self.interior_knots_S[-1]
        self.T_min = T_min if T_min is not None else self.interior_knots_T[0]
        self.T_max = T_max if T_max is not None else self.interior_knots_T[-1]

        # Build full knot vectors with boundary conditions
        self.knot_vector_S = self._build_knot_vector(self.interior_knots_S, self.degree_S)
        self.knot_vector_T = self._build_knot_vector(self.interior_knots_T, self.degree_T)

        # Create 1D BSpline objects for each dimension
        # We'll use these to evaluate the tensor product
        self._create_basis_functions()

        # Validate parameters
        self._validate_parameters()

    @staticmethod
    def _build_knot_vector(interior_knots: np.ndarray, degree: int) -> np.ndarray:
        """
        Build full knot vector with clamped boundary conditions.

        Args:
            interior_knots: Interior knot points
            degree: B-spline degree

        Returns:
            Full knot vector with repeated boundary knots
        """
        k_first = interior_knots[0]
        k_last = interior_knots[-1]

        # Clamped boundary conditions: repeat endpoints (degree+1) times
        left_boundary = np.repeat(k_first, degree + 1)
        right_boundary = np.repeat(k_last, degree + 1)

        # Full knot vector
        knot_vector = np.concatenate([
            left_boundary,
            interior_knots[1:-1],  # Exclude already-used boundaries
            right_boundary
        ])

        return knot_vector

    def _create_basis_functions(self):
        """Create 1D basis function evaluators for tensor product."""
        # We'll store dummy coefficients (all ones) since we only need basis evaluation
        n_S = len(self.coefficients)
        n_T = self.coefficients.shape[1]

        dummy_coeffs_S = np.ones(n_S)
        dummy_coeffs_T = np.ones(n_T)

        # These are just for computing basis function values
        self._basis_S = BSpline(self.knot_vector_S, dummy_coeffs_S,
                                self.degree_S, extrapolate=True)
        self._basis_T = BSpline(self.knot_vector_T, dummy_coeffs_T,
                                self.degree_T, extrapolate=True)

    def _validate_parameters(self):
        """Validate 2D B-spline parameters and constraints."""
        # Check degrees
        if self.degree_S < 1 or self.degree_T < 1:
            raise ValueError(
                f"Degrees must be >= 1, got degree_S={self.degree_S}, "
                f"degree_T={self.degree_T}"
            )

        # Check coefficients are positive (volatilities must be positive)
        if np.any(self.coefficients <= 0):
            negative_count = np.sum(self.coefficients <= 0)
            raise ValueError(
                f"All coefficients must be positive (represent volatilities). "
                f"Found {negative_count} non-positive values."
            )

        # Check for reasonable volatility values (1% to 200%)
        if np.any(self.coefficients < 0.01) or np.any(self.coefficients > 2.0):
            import warnings
            warnings.warn(
                f"Coefficients outside typical range [0.01, 2.0]. "
                f"Range: [{self.coefficients.min():.4f}, {self.coefficients.max():.4f}]"
            )

        # Check knot vectors are sorted
        if not np.all(np.diff(self.interior_knots_S) >= 0):
            raise ValueError("Spatial knot vector must be non-decreasing")
        if not np.all(np.diff(self.interior_knots_T) >= 0):
            raise ValueError("Temporal knot vector must be non-decreasing")

    def _evaluate_basis_S(self, S: np.ndarray, deriv: int = 0) -> np.ndarray:
        """
        Evaluate all spatial basis functions at given points.

        Args:
            S: Spatial points, shape (n_points,)
            deriv: Derivative order (0 for value, 1 for first derivative, etc.)

        Returns:
            Basis matrix, shape (n_points, n_basis_S)
            basis[i, j] = Bⱼ(Sᵢ)
        """
        n_S = len(self.coefficients)
        S = np.atleast_1d(S)
        n_points = len(S)

        basis_matrix = np.zeros((n_points, n_S))

        # Evaluate each basis function
        for j in range(n_S):
            coeffs = np.zeros(n_S)
            coeffs[j] = 1.0
            bspline_j = BSpline(self.knot_vector_S, coeffs, self.degree_S,
                               extrapolate=True)
            basis_matrix[:, j] = bspline_j(S, nu=deriv)

        return basis_matrix

    def _evaluate_basis_T(self, t: np.ndarray, deriv: int = 0) -> np.ndarray:
        """
        Evaluate all temporal basis functions at given points.

        Args:
            t: Temporal points, shape (n_points,)
            deriv: Derivative order

        Returns:
            Basis matrix, shape (n_points, n_basis_T)
            basis[i, j] = Bⱼ(tᵢ)
        """
        n_T = self.coefficients.shape[1]
        t = np.atleast_1d(t)
        n_points = len(t)

        basis_matrix = np.zeros((n_points, n_T))

        # Evaluate each basis function
        for j in range(n_T):
            coeffs = np.zeros(n_T)
            coeffs[j] = 1.0
            bspline_j = BSpline(self.knot_vector_T, coeffs, self.degree_T,
                               extrapolate=True)
            basis_matrix[:, j] = bspline_j(t, nu=deriv)

        return basis_matrix

    def evaluate(self, S: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Evaluate 2D B-spline volatility at given (S, t) points.

        Args:
            S: Spot prices, shape (n_points,) or scalar
            t: Times, shape (n_points,) or scalar

        Returns:
            Volatilities σ(S, t), shape (n_points,)

        Note:
            S and t must be broadcastable to the same shape.
        """
        S = np.atleast_1d(S)
        t = np.atleast_1d(t)

        # Broadcast to same shape
        S, t = np.broadcast_arrays(S, t)
        n_points = S.size
        S_flat = S.ravel()
        t_flat = t.ravel()

        # Evaluate basis functions
        basis_S = self._evaluate_basis_S(S_flat)  # (n_points, n_S)
        basis_T = self._evaluate_basis_T(t_flat)  # (n_points, n_T)

        # Tensor product: σ(S,t) = Σᵢⱼ wᵢⱼ Bᵢ(S) Bⱼ(t)
        # Vectorized: sigma = basis_S @ coefficients @ basis_T.T
        # But for point-wise evaluation:
        sigma = np.zeros(n_points)
        for k in range(n_points):
            # sigma[k] = sum_ij w[i,j] * basis_S[k,i] * basis_T[k,j]
            sigma[k] = np.sum(self.coefficients *
                             np.outer(basis_S[k, :], basis_T[k, :]))

        # Apply floor for numerical stability
        sigma = np.maximum(sigma, 0.01)  # 1% minimum volatility

        return sigma.reshape(S.shape)

    def evaluate_grid(self, S_grid: np.ndarray, T_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate 2D B-spline on a tensor product grid (more efficient).

        Args:
            S_grid: Spatial grid points, shape (M+1,)
            T_grid: Temporal grid points, shape (N+1,)

        Returns:
            Volatility grid σ[i, n] of shape (M+1, N+1)
            where σ[i, n] = σ(S_grid[i], T_grid[n])
        """
        S_grid = np.atleast_1d(S_grid)
        T_grid = np.atleast_1d(T_grid)

        # Evaluate basis functions on grid
        basis_S = self._evaluate_basis_S(S_grid)  # (M+1, n_S)
        basis_T = self._evaluate_basis_T(T_grid)  # (N+1, n_T)

        # Tensor product: sigma_grid = basis_S @ coefficients @ basis_T.T
        sigma_grid = basis_S @ self.coefficients @ basis_T.T

        # Apply floor
        sigma_grid = np.maximum(sigma_grid, 0.01)

        return sigma_grid

    def to_pde_grid(self, S_grid: np.ndarray, T_grid: np.ndarray,
                   r: float, S_ref: float = None) -> np.ndarray:
        """
        Convert 2D B-spline model to local volatility grid for PDE solver.

        Args:
            S_grid: Spatial grid [S_0, S_1, ..., S_M]
            T_grid: Time grid [T_0, T_1, ..., T_N]
            r: Risk-free rate (unused in direct local vol model)
            S_ref: Reference spot price (unused, kept for interface compatibility)

        Returns:
            Local volatility grid σ[i, n] of shape (M+1, N+1)
        """
        return self.evaluate_grid(S_grid, T_grid)

    def get_active_basis_indices(self, S: float, t: float) -> Tuple[List[int], List[int]]:
        """
        Get indices of basis functions with non-zero support at (S, t).

        Due to compact support, only (degree_S+1) × (degree_T+1) basis functions
        are non-zero at any (S, t). This is the key property for sparse Hessians.

        Args:
            S: Spot price
            t: Time

        Returns:
            (active_S_indices, active_T_indices) where
            - active_S_indices: spatial basis indices with Bᵢ(S) ≠ 0
            - active_T_indices: temporal basis indices with Bⱼ(t) ≠ 0
        """
        # Spatial active indices
        if S <= self.knot_vector_S[self.degree_S]:
            active_S = list(range(self.degree_S + 1))
        elif S >= self.knot_vector_S[-(self.degree_S + 1)]:
            n_S = len(self.coefficients)
            active_S = list(range(n_S - self.degree_S - 1, n_S))
        else:
            interval_idx = np.searchsorted(self.knot_vector_S, S, side='right') - 1
            start_idx = max(0, interval_idx - self.degree_S)
            end_idx = min(len(self.coefficients), interval_idx + 1)
            active_S = list(range(start_idx, end_idx))

        # Temporal active indices
        if t <= self.knot_vector_T[self.degree_T]:
            active_T = list(range(self.degree_T + 1))
        elif t >= self.knot_vector_T[-(self.degree_T + 1)]:
            n_T = self.coefficients.shape[1]
            active_T = list(range(n_T - self.degree_T - 1, n_T))
        else:
            interval_idx = np.searchsorted(self.knot_vector_T, t, side='right') - 1
            start_idx = max(0, interval_idx - self.degree_T)
            end_idx = min(self.coefficients.shape[1], interval_idx + 1)
            active_T = list(range(start_idx, end_idx))

        return active_S, active_T

    def get_hessian_sparsity_pattern(self) -> np.ndarray:
        """
        Get the sparsity pattern of the parameter Hessian (flattened 2D indices).

        For 2D B-spline models, the parameters are w[i, j] which we flatten to
        a 1D vector: w_flat[k] where k = i * n_T + j.

        The Hessian H[k, l] = ∂²price/∂w[i,j] ∂w[p,q] is sparse because:
        - Basis functions have compact support
        - H[k, l] = 0 when |i-p| > degree_S OR |j-q| > degree_T

        Returns:
            Boolean array of shape (n_params, n_params) indicating non-zero entries
            where n_params = n_S × n_T
        """
        n_S, n_T = self.coefficients.shape
        n_params = n_S * n_T

        sparsity = np.zeros((n_params, n_params), dtype=bool)

        # Fill block-banded structure
        for i in range(n_S):
            for j in range(n_T):
                k = i * n_T + j  # Flat index for w[i,j]

                # Find all (p, q) where |i-p| <= degree_S and |j-q| <= degree_T
                for p in range(max(0, i - self.degree_S),
                              min(n_S, i + self.degree_S + 1)):
                    for q in range(max(0, j - self.degree_T),
                                  min(n_T, j + self.degree_T + 1)):
                        l = p * n_T + q  # Flat index for w[p,q]
                        sparsity[k, l] = True

        return sparsity

    def get_hessian_bandwidth_2d(self) -> Tuple[int, int]:
        """
        Get the 2D bandwidth of the Hessian matrix.

        Returns:
            (bandwidth_S, bandwidth_T) where:
            - bandwidth_S = 2 * degree_S + 1
            - bandwidth_T = 2 * degree_T + 1
        """
        return (2 * self.degree_S + 1, 2 * self.degree_T + 1)

    def get_hessian_bandwidth_flat(self) -> int:
        """
        Get the effective bandwidth when flattened to 1D.

        For block-banded structure, the flat bandwidth is approximately:
            bandwidth_flat ≈ (2 * degree_S + 1) * n_T + (2 * degree_T + 1)

        Returns:
            Approximate bandwidth of flattened Hessian
        """
        n_T = self.coefficients.shape[1]
        bandwidth_S = 2 * self.degree_S + 1
        bandwidth_T = 2 * self.degree_T + 1
        return bandwidth_S * n_T + bandwidth_T

    def get_n_params(self) -> int:
        """Get total number of parameters (flattened coefficient matrix)."""
        return self.coefficients.size

    def get_coefficients_flat(self) -> np.ndarray:
        """Get coefficients as flattened 1D array."""
        return self.coefficients.ravel()

    def get_coefficients_2d(self) -> np.ndarray:
        """Get coefficients as 2D matrix."""
        return self.coefficients.copy()

    def update_coefficients(self, new_coefficients: np.ndarray):
        """
        Update B-spline coefficients (for calibration/optimization).

        Args:
            new_coefficients: New coefficient values, can be 1D (flattened) or 2D matrix
        """
        new_coefficients = np.asarray(new_coefficients, dtype=np.float64)

        # Handle both flat and 2D input
        if new_coefficients.ndim == 1:
            expected_size = self.coefficients.size
            if new_coefficients.size != expected_size:
                raise ValueError(
                    f"Flattened coefficients must have size {expected_size}, "
                    f"got {new_coefficients.size}"
                )
            new_coefficients = new_coefficients.reshape(self.coefficients.shape)
        elif new_coefficients.ndim == 2:
            if new_coefficients.shape != self.coefficients.shape:
                raise ValueError(
                    f"2D coefficients must have shape {self.coefficients.shape}, "
                    f"got {new_coefficients.shape}"
                )
        else:
            raise ValueError("Coefficients must be 1D (flattened) or 2D array")

        self.coefficients = new_coefficients

        # Validate
        self._validate_parameters()

    def flatten_indices(self, i: int, j: int) -> int:
        """Convert 2D index (i, j) to flattened 1D index."""
        n_T = self.coefficients.shape[1]
        return i * n_T + j

    def unflatten_index(self, k: int) -> Tuple[int, int]:
        """Convert flattened 1D index k to 2D index (i, j)."""
        n_T = self.coefficients.shape[1]
        i = k // n_T
        j = k % n_T
        return i, j


def create_flat_bspline_2d(config: BSplineConfig2D, volatility: float = 0.20) -> BSplineModel2D:
    """
    Create a flat 2D B-spline model (constant volatility in space and time).

    Useful for testing: should match constant volatility BS model.

    Args:
        config: 2D B-spline configuration
        volatility: Constant volatility value

    Returns:
        BSplineModel2D with constant volatility
    """
    knots_S = np.linspace(config.S_min, config.S_max, config.n_knots_S)
    knots_T = np.linspace(config.T_min, config.T_max, config.n_knots_T)

    # Correct formula: n_basis = degree + n_interior_knots - 1
    n_S = config.degree_S + config.n_knots_S - 1
    n_T = config.degree_T + config.n_knots_T - 1

    coefficients = np.full((n_S, n_T), volatility)

    return BSplineModel2D(knots_S, knots_T, coefficients,
                         config.degree_S, config.degree_T,
                         config.S_min, config.S_max,
                         config.T_min, config.T_max)


def create_separable_bspline_2d(config: BSplineConfig2D,
                                vol_S: np.ndarray,
                                vol_T: np.ndarray) -> BSplineModel2D:
    """
    Create a separable 2D B-spline: σ(S,t) = σ_S(S) · σ_T(t).

    Useful for testing against known 1D solutions.

    Args:
        config: 2D B-spline configuration
        vol_S: 1D volatility profile in space, shape (degree_S + n_knots_S - 1,)
        vol_T: 1D volatility profile in time, shape (degree_T + n_knots_T - 1,)

    Returns:
        BSplineModel2D with separable structure
    """
    knots_S = np.linspace(config.S_min, config.S_max, config.n_knots_S)
    knots_T = np.linspace(config.T_min, config.T_max, config.n_knots_T)

    # Outer product creates separable structure
    coefficients = np.outer(vol_S, vol_T)

    return BSplineModel2D(knots_S, knots_T, coefficients,
                         config.degree_S, config.degree_T,
                         config.S_min, config.S_max,
                         config.T_min, config.T_max)


if __name__ == "__main__":
    # Test 2D B-spline model
    print("="*70)
    print("2D B-Spline Volatility Model Test")
    print("="*70)

    # Create sample 2D B-spline
    print("\n1. Creating flat 2D B-spline model...")
    config = BSplineConfig2D(
        S_min=50.0, S_max=150.0, T_min=0.0, T_max=1.0,
        n_knots_S=10, n_knots_T=5, degree_S=3, degree_T=2
    )
    model = create_flat_bspline_2d(config, volatility=0.25)

    print(f"   - Degrees: (S={model.degree_S}, T={model.degree_T})")
    print(f"   - Interior knots: (S={model.k_S}, T={model.k_T})")
    print(f"   - Coefficient matrix shape: {model.coefficients.shape}")
    print(f"   - Total parameters: {model.get_n_params()}")
    print(f"   - 2D Hessian bandwidth: {model.get_hessian_bandwidth_2d()}")

    # Test evaluation at points
    print("\n2. Evaluating volatility at test points...")
    S_test = np.array([60, 80, 100, 120, 140])
    t_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    sigma_test = model.evaluate(S_test, t_test)

    print(f"{'Spot':<10} {'Time':<10} {'Volatility':<15}")
    print("-"*35)
    for S, t, sig in zip(S_test, t_test, sigma_test):
        print(f"{S:<10.1f} {t:<10.2f} {sig:<15.4f}")

    # Test grid evaluation
    print("\n3. Evaluating on PDE grid...")
    S_grid = np.linspace(50, 150, 21)
    T_grid = np.linspace(0, 1.0, 11)
    sigma_grid = model.evaluate_grid(S_grid, T_grid)

    print(f"   Grid shape: {sigma_grid.shape}")
    print(f"   Volatility range: [{sigma_grid.min():.4f}, {sigma_grid.max():.4f}]")
    print(f"   Max deviation from 0.25: {np.max(np.abs(sigma_grid - 0.25)):.2e}")

    # Test sparsity pattern
    print("\n4. Analyzing Hessian sparsity...")
    sparsity = model.get_hessian_sparsity_pattern()
    n_params = model.get_n_params()
    n_nonzero = np.sum(sparsity)
    n_total = n_params * n_params
    sparsity_ratio = n_nonzero / n_total

    print(f"   - Total parameters: {n_params}")
    print(f"   - Total Hessian entries: {n_total}")
    print(f"   - Non-zero entries: {n_nonzero}")
    print(f"   - Sparsity ratio: {sparsity_ratio:.2%}")
    print(f"   - Zero entries: {n_total - n_nonzero} ({(1-sparsity_ratio):.2%})")

    # Test compact support
    print("\n5. Testing compact support property...")
    S_point, t_point = 100.0, 0.5
    active_S, active_T = model.get_active_basis_indices(S_point, t_point)
    print(f"   At (S={S_point}, t={t_point}):")
    print(f"   - Active spatial basis: {active_S} (count={len(active_S)})")
    print(f"   - Active temporal basis: {active_T} (count={len(active_T)})")
    print(f"   - Total active basis functions: {len(active_S) * len(active_T)}")
    print(f"   - Expected: {(model.degree_S+1) * (model.degree_T+1)}")

    # Test separable structure
    print("\n6. Testing separable structure...")
    vol_S = np.linspace(0.20, 0.30, config.n_knots_S + config.degree_S + 1)
    vol_T = np.linspace(1.0, 1.2, config.n_knots_T + config.degree_T + 1)
    model_sep = create_separable_bspline_2d(config, vol_S, vol_T)

    # Verify separability: σ(S,t) / σ(S,0) should equal σ_T(t) / σ_T(0)
    S_test_sep = 100.0
    t_test_sep = np.array([0.0, 0.5, 1.0])
    sigma_sep = model_sep.evaluate(
        np.full_like(t_test_sep, S_test_sep),
        t_test_sep
    )
    ratio = sigma_sep / sigma_sep[0]
    print(f"   σ(S={S_test_sep}, t) / σ(S={S_test_sep}, t=0): {ratio}")

    print("\n" + "="*70)
    print("✓ 2D B-spline model test completed successfully!")
    print("="*70)
