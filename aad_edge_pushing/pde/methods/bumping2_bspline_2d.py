"""
Bumping2 Method for 2D B-Spline Volatility Models

Pure finite difference method for computing Hessian ∂²V/∂wᵢⱼ∂wₖₗ with respect
to 2D B-spline coefficients.

For n_S × n_T parameters, full Hessian requires O(n²) PDE solves, which is
computationally prohibitive. This implementation provides:

1. **Diagonal Hessian** (default): ∂²V/∂wᵢⱼ² for all parameters
   - Requires: 2 × n_S × n_T + 1 PDE solves
   - Used for: Validation of edge-pushing diagonal

2. **Sampled Cross-Derivatives**: ∂²V/∂wᵢⱼ∂wₖₗ for selected pairs
   - Requires: ~4 PDE solves per pair
   - Used for: Spot-checking edge-pushing off-diagonal accuracy

3. **Gradient** (Jacobian): ∂V/∂wᵢⱼ via central differences
   - Requires: 2 × n_S × n_T PDE solves
   - Used for: First-order sensitivity validation

Unified Grid Strategy:
All PDE solves use the same S_max based on the maximum volatility across all
coefficient perturbations. This is CRITICAL for accurate second derivatives.

Reference:
This extends the 2-parameter Bumping2 (S0, σ) to high-dimensional B-spline
coefficient space (wᵢⱼ).
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class Bumping2BSpline2D:
    """
    Finite difference Hessian computation for 2D B-spline volatility models.

    This is the validation method for edge-pushing. It's slow but accurate.
    """

    def __init__(self,
                 bspline_model_2d,
                 S0: float,
                 K: float,
                 T: float,
                 r: float,
                 M: int = 200,
                 N: int = 50,
                 eps_coeff: float = 0.001,
                 use_adaptive_Smax: bool = True,
                 sigma_margin: float = 0.1):
        """
        Initialize Bumping2 method for 2D B-spline models.

        Args:
            bspline_model_2d: BSplineModel2D instance
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            M: Number of spatial grid points
            N: Number of time steps
            eps_coeff: Finite difference step size for coefficients
                      (relative to coefficient value, default 0.1%)
            use_adaptive_Smax: Use volatility-adaptive S_max
            sigma_margin: Safety margin for grid sizing
        """
        from aad_edge_pushing.pde.models.bspline_model_2d import BSplineModel2D

        if not isinstance(bspline_model_2d, BSplineModel2D):
            raise TypeError("bspline_model_2d must be a BSplineModel2D instance")

        self.model = bspline_model_2d
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.M = M
        self.N = N
        self.eps_coeff = eps_coeff
        self.use_adaptive_Smax = use_adaptive_Smax
        self.sigma_margin = sigma_margin

        # Get model parameters
        self.n_S, self.n_T = self.model.coefficients.shape
        self.n_params = self.n_S * self.n_T

    def _compute_unified_Smax(self, coeff_matrix: np.ndarray) -> float:
        """
        Compute unified S_max for all perturbations.

        Uses the maximum volatility that could occur across all coefficient
        perturbations to ensure consistent grid sizing.

        IMPORTANT: Must match the S_max computation in BS_PDE_AAD_BSpline2D
        to ensure consistent grids between edge-pushing and bumping2.
        """
        from aad_edge_pushing.pde.pde_config import PDEConfig

        # Match BS_PDE_AAD_BSpline2D behavior
        if self.use_adaptive_Smax:
            # Sample volatility at S0 across time
            t_samples = np.linspace(0, self.T, 10)
            S_samples = np.full_like(t_samples, self.S0)

            # Create temporary model to evaluate max volatility
            temp_model = self._create_temp_model(coeff_matrix)
            sigma_samples = temp_model.evaluate(S_samples, t_samples)

            # Get base volatility
            sigma_center = np.max(sigma_samples)

            # Compute S_max with SAME margin as Edge-Pushing for consistency
            # IMPORTANT: Use only sigma_margin (NOT eps_coeff + sigma_margin)
            # to match Edge-Pushing exactly
            S_max = PDEConfig.compute_unified_Smax(
                self.K, self.T, sigma_center, self.r,
                sigma_margin=self.sigma_margin  # Match Edge-Pushing exactly
            )
        else:
            # Simple fixed multiplier (matches BS_PDE_AAD_BSpline2D line 87)
            S_max = 5.0 * self.K

        return S_max

    def _create_temp_model(self, coeff_matrix: np.ndarray):
        """Create temporary model with given coefficients."""
        from aad_edge_pushing.pde.models.bspline_model_2d import BSplineModel2D

        # Create a copy with new coefficients
        temp_model = BSplineModel2D(
            knots_S=self.model.interior_knots_S,
            knots_T=self.model.interior_knots_T,
            coefficients=coeff_matrix.copy(),
            degree_S=self.model.degree_S,
            degree_T=self.model.degree_T,
            S_min=self.model.S_min,
            S_max=self.model.S_max,
            T_min=self.model.T_min,
            T_max=self.model.T_max
        )
        return temp_model

    def _solve_pde_simple(self, coeff_matrix: np.ndarray, S_max: float) -> float:
        """
        Solve PDE with given coefficients using simple numerical solver.

        Args:
            coeff_matrix: 2D coefficient matrix
            S_max: Fixed S_max for consistency

        Returns:
            Option price at S0
        """
        from aad_edge_pushing.pde.simple_pde_solver import SimplePDESolver

        # Create temporary model
        temp_model = self._create_temp_model(coeff_matrix)

        # Create PDE solver
        # IMPORTANT: Use n_rannacher=0 to match Edge-Pushing (pure Crank-Nicolson)
        solver = SimplePDESolver(
            S0=self.S0,
            K=self.K,
            T=self.T,
            r=self.r,
            sigma=0.25,  # Not used (will use volatility_model)
            M=self.M,
            N_base=self.N,
            volatility_model=temp_model,  # Pass 2D model
            S_max_override=S_max,
            n_rannacher=0  # No Rannacher steps - pure Crank-Nicolson like Edge-Pushing
        )

        # Solve PDE
        price, _ = solver._solve_pde_numerical(self.S0, sigma=0.25)

        return price

    def compute_gradient(self, coeff_matrix: np.ndarray,
                        verbose: bool = False) -> Dict:
        """
        Compute gradient ∂V/∂wᵢⱼ via central differences.

        Args:
            coeff_matrix: 2D coefficient matrix (n_S, n_T)
            verbose: Print progress

        Returns:
            Dictionary with:
                - gradient: Flattened gradient vector
                - price: Base option price
                - n_pde_solves: Number of PDE solves
                - computation_time_ms: Wall-clock time
        """
        t_start = time.perf_counter()

        if verbose:
            print(f"  Computing gradient for {self.n_params} parameters...")

        # Compute unified S_max
        S_max = self._compute_unified_Smax(coeff_matrix)

        if verbose:
            print(f"  Unified S_max: {S_max:.2f}")

        # Base price
        V0 = self._solve_pde_simple(coeff_matrix, S_max)

        # Gradient via central differences
        gradient = np.zeros((self.n_S, self.n_T))

        for i in range(self.n_S):
            for j in range(self.n_T):
                if verbose and (i * self.n_T + j) % 5 == 0:
                    print(f"    Parameter {i},{j} ({i*self.n_T+j+1}/{self.n_params})")

                # Perturb coefficient wᵢⱼ
                eps = self.eps_coeff * coeff_matrix[i, j]

                # Forward perturbation
                coeff_p = coeff_matrix.copy()
                coeff_p[i, j] += eps
                V_p = self._solve_pde_simple(coeff_p, S_max)

                # Backward perturbation
                coeff_m = coeff_matrix.copy()
                coeff_m[i, j] -= eps
                V_m = self._solve_pde_simple(coeff_m, S_max)

                # Central difference
                gradient[i, j] = (V_p - V_m) / (2 * eps)

        time_ms = (time.perf_counter() - t_start) * 1000

        if verbose:
            print(f"  ✓ Gradient computed in {time_ms/1000:.1f}s")
            print(f"  Gradient norm: {np.linalg.norm(gradient):.6e}")

        return {
            'gradient': gradient.ravel(),
            'gradient_2d': gradient,
            'price': V0,
            'n_pde_solves': 1 + 2 * self.n_params,
            'computation_time_ms': time_ms
        }

    def compute_diagonal_hessian(self, coeff_matrix: np.ndarray,
                                verbose: bool = False) -> Dict:
        """
        Compute diagonal Hessian ∂²V/∂wᵢⱼ² for all parameters.

        Uses central difference formula:
            ∂²V/∂wᵢⱼ² ≈ [V(wᵢⱼ+ε) - 2V(wᵢⱼ) + V(wᵢⱼ-ε)] / ε²

        Args:
            coeff_matrix: 2D coefficient matrix (n_S, n_T)
            verbose: Print progress

        Returns:
            Dictionary with:
                - hessian_diagonal: Diagonal of Hessian (flattened)
                - gradient: Gradient vector (from side computation)
                - price: Base option price
                - n_pde_solves: Number of PDE solves
                - computation_time_ms: Wall-clock time
        """
        t_start = time.perf_counter()

        if verbose:
            print(f"  Computing diagonal Hessian for {self.n_params} parameters...")

        # Compute unified S_max
        S_max = self._compute_unified_Smax(coeff_matrix)

        if verbose:
            print(f"  Unified S_max: {S_max:.2f}")

        # Base price
        V0 = self._solve_pde_simple(coeff_matrix, S_max)

        # Diagonal Hessian and gradient
        hessian_diag = np.zeros((self.n_S, self.n_T))
        gradient = np.zeros((self.n_S, self.n_T))

        for i in range(self.n_S):
            for j in range(self.n_T):
                if verbose and (i * self.n_T + j) % 5 == 0:
                    print(f"    Parameter {i},{j} ({i*self.n_T+j+1}/{self.n_params})")

                # Perturbation size
                eps = self.eps_coeff * coeff_matrix[i, j]

                # Forward perturbation
                coeff_p = coeff_matrix.copy()
                coeff_p[i, j] += eps
                V_p = self._solve_pde_simple(coeff_p, S_max)

                # Backward perturbation
                coeff_m = coeff_matrix.copy()
                coeff_m[i, j] -= eps
                V_m = self._solve_pde_simple(coeff_m, S_max)

                # Central difference for gradient
                gradient[i, j] = (V_p - V_m) / (2 * eps)

                # Second derivative
                hessian_diag[i, j] = (V_p - 2*V0 + V_m) / (eps**2)

        time_ms = (time.perf_counter() - t_start) * 1000

        if verbose:
            print(f"  ✓ Diagonal Hessian computed in {time_ms/1000:.1f}s")
            print(f"  Gradient norm: {np.linalg.norm(gradient):.6e}")
            print(f"  Hessian diagonal range: [{hessian_diag.min():.6e}, "
                  f"{hessian_diag.max():.6e}]")

        return {
            'hessian_diagonal': hessian_diag.ravel(),
            'hessian_diagonal_2d': hessian_diag,
            'gradient': gradient.ravel(),
            'gradient_2d': gradient,
            'price': V0,
            'n_pde_solves': 1 + 2 * self.n_params,
            'computation_time_ms': time_ms
        }

    def compute_cross_derivative(self, coeff_matrix: np.ndarray,
                                 i1: int, j1: int, i2: int, j2: int,
                                 S_max: Optional[float] = None) -> float:
        """
        Compute single cross-derivative ∂²V/∂wᵢ₁ⱼ₁∂wᵢ₂ⱼ₂.

        Uses finite difference of finite differences:
            ∂²V/∂w₁∂w₂ ≈ [∂V/∂w₁(w₂+ε) - ∂V/∂w₁(w₂-ε)] / (2ε)

        Args:
            coeff_matrix: 2D coefficient matrix
            i1, j1: First parameter indices
            i2, j2: Second parameter indices
            S_max: Unified S_max (computed if None)

        Returns:
            Cross-derivative value
        """
        if S_max is None:
            S_max = self._compute_unified_Smax(coeff_matrix)

        eps1 = self.eps_coeff * coeff_matrix[i1, j1]
        eps2 = self.eps_coeff * coeff_matrix[i2, j2]

        # Four corner points
        coeff_pp = coeff_matrix.copy()
        coeff_pp[i1, j1] += eps1
        coeff_pp[i2, j2] += eps2
        V_pp = self._solve_pde_simple(coeff_pp, S_max)

        coeff_pm = coeff_matrix.copy()
        coeff_pm[i1, j1] += eps1
        coeff_pm[i2, j2] -= eps2
        V_pm = self._solve_pde_simple(coeff_pm, S_max)

        coeff_mp = coeff_matrix.copy()
        coeff_mp[i1, j1] -= eps1
        coeff_mp[i2, j2] += eps2
        V_mp = self._solve_pde_simple(coeff_mp, S_max)

        coeff_mm = coeff_matrix.copy()
        coeff_mm[i1, j1] -= eps1
        coeff_mm[i2, j2] -= eps2
        V_mm = self._solve_pde_simple(coeff_mm, S_max)

        # Cross-derivative via finite difference of finite differences
        cross_deriv = (V_pp - V_pm - V_mp + V_mm) / (4 * eps1 * eps2)

        return cross_deriv

    def compute_sampled_cross_derivatives(self,
                                         coeff_matrix: np.ndarray,
                                         sample_pairs: List[Tuple[Tuple[int, int],
                                                                  Tuple[int, int]]],
                                         verbose: bool = False) -> Dict:
        """
        Compute cross-derivatives for selected parameter pairs.

        Args:
            coeff_matrix: 2D coefficient matrix
            sample_pairs: List of ((i1, j1), (i2, j2)) pairs
            verbose: Print progress

        Returns:
            Dictionary with cross-derivatives and metadata
        """
        t_start = time.perf_counter()

        if verbose:
            print(f"  Computing {len(sample_pairs)} cross-derivatives...")

        S_max = self._compute_unified_Smax(coeff_matrix)

        cross_derivs = {}
        for idx, ((i1, j1), (i2, j2)) in enumerate(sample_pairs):
            if verbose:
                print(f"    Pair {idx+1}/{len(sample_pairs)}: "
                      f"w[{i1},{j1}] × w[{i2},{j2}]")

            cross_deriv = self.compute_cross_derivative(
                coeff_matrix, i1, j1, i2, j2, S_max
            )

            # Store with both orderings (Hessian is symmetric)
            cross_derivs[(i1, j1, i2, j2)] = cross_deriv
            cross_derivs[(i2, j2, i1, j1)] = cross_deriv

        time_ms = (time.perf_counter() - t_start) * 1000

        if verbose:
            print(f"  ✓ Cross-derivatives computed in {time_ms/1000:.1f}s")

        return {
            'cross_derivatives': cross_derivs,
            'sample_pairs': sample_pairs,
            'n_pde_solves': 4 * len(sample_pairs),
            'computation_time_ms': time_ms
        }


def suggest_sample_pairs_2d(n_S: int, n_T: int,
                           n_pairs: int = 10,
                           strategy: str = 'neighbors') -> List[Tuple[Tuple[int, int],
                                                                      Tuple[int, int]]]:
    """
    Suggest parameter pairs to sample for cross-derivative validation.

    Args:
        n_S, n_T: Coefficient matrix dimensions
        n_pairs: Number of pairs to sample
        strategy: Sampling strategy
            - 'neighbors': Sample adjacent parameters (should have nonzero Hessian)
            - 'diagonal': Sample along diagonal (i,j) × (i+1,j+1)
            - 'random': Random pairs

    Returns:
        List of ((i1, j1), (i2, j2)) pairs
    """
    pairs = []

    if strategy == 'neighbors':
        # Sample spatial neighbors: (i,j) × (i+1,j)
        for _ in range(min(n_pairs // 2, n_S - 1)):
            i = np.random.randint(0, n_S - 1)
            j = np.random.randint(0, n_T)
            pairs.append(((i, j), (i+1, j)))

        # Sample temporal neighbors: (i,j) × (i,j+1)
        for _ in range(n_pairs - len(pairs)):
            i = np.random.randint(0, n_S)
            j = np.random.randint(0, n_T - 1)
            pairs.append(((i, j), (i, j+1)))

    elif strategy == 'diagonal':
        # Sample diagonal pairs
        for _ in range(n_pairs):
            i = np.random.randint(0, n_S - 1)
            j = np.random.randint(0, n_T - 1)
            pairs.append(((i, j), (i+1, j+1)))

    elif strategy == 'random':
        # Completely random pairs
        for _ in range(n_pairs):
            i1, j1 = np.random.randint(0, n_S), np.random.randint(0, n_T)
            i2, j2 = np.random.randint(0, n_S), np.random.randint(0, n_T)
            if (i1, j1) != (i2, j2):
                pairs.append(((i1, j1), (i2, j2)))

    return pairs


if __name__ == "__main__":
    from aad_edge_pushing.pde.models.bspline_model_2d import (
        BSplineModel2D, BSplineConfig2D, create_flat_bspline_2d
    )

    print("="*70)
    print("Bumping2 for 2D B-Spline - Diagonal Hessian Test")
    print("="*70)

    # Create minimal model
    print("\n1. Creating minimal 2D B-spline...")
    config = BSplineConfig2D(
        S_min=80.0, S_max=120.0, T_min=0.0, T_max=1.0,
        n_knots_S=2, n_knots_T=2,
        degree_S=2, degree_T=1
    )
    model = create_flat_bspline_2d(config, volatility=0.25)
    print(f"   Parameters: {model.get_n_params()} (shape: {model.coefficients.shape})")

    # Create Bumping2 method
    print("\n2. Creating Bumping2 method...")
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    bumping2 = Bumping2BSpline2D(
        model, S0, K, T, r,
        M=50, N=10,  # Small grid for testing
        eps_coeff=0.01
    )

    # Compute diagonal Hessian
    print("\n3. Computing diagonal Hessian...")
    coeff_matrix = model.get_coefficients_2d()
    result = bumping2.compute_diagonal_hessian(coeff_matrix, verbose=True)

    print("\n4. Results:")
    print(f"   Price: {result['price']:.6f}")
    print(f"   PDE solves: {result['n_pde_solves']}")
    print(f"   Time: {result['computation_time_ms']/1000:.1f}s")
    print(f"   Gradient norm: {np.linalg.norm(result['gradient']):.6e}")
    print(f"   Hessian diagonal range: "
          f"[{result['hessian_diagonal'].min():.6e}, "
          f"{result['hessian_diagonal'].max():.6e}]")

    # Sample cross-derivatives
    print("\n5. Sampling cross-derivatives...")
    pairs = suggest_sample_pairs_2d(model.coefficients.shape[0],
                                   model.coefficients.shape[1],
                                   n_pairs=3, strategy='neighbors')
    cross_result = bumping2.compute_sampled_cross_derivatives(
        coeff_matrix, pairs, verbose=True
    )

    print("\n   Cross-derivatives:")
    for (i1, j1, i2, j2), val in cross_result['cross_derivatives'].items():
        if i1 <= i2 and j1 <= j2:  # Print each pair once
            print(f"     ∂²V/∂w[{i1},{j1}]∂w[{i2},{j2}] = {val:.6e}")

    print("\n" + "="*70)
    print("✓ Bumping2 2D test completed!")
    print("="*70)
