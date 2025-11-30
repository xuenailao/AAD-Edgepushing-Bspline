"""
B-Spline Volatility Calibration from Market Data

This module implements calibration of B-spline volatility parameters to match
market-observed implied volatilities. The calibration problem is:

    min_{w} Σᵢ (σ_market(Kᵢ) - σ_model(Kᵢ; w))² + λ·R(w)

where:
- w = [w₀, w₁, ..., wₙ]: B-spline coefficients
- σ_market(Kᵢ): Market implied volatilities at strikes Kᵢ
- σ_model(Kᵢ; w): B-spline volatility σ(S) = Σⱼ wⱼ·Bⱼ(S)
- R(w): Regularization term (Tikhonov/roughness penalty)
- λ: Regularization weight

Key Features:
1. **Gradient-based optimization**: Use edge-pushing to compute ∂Loss/∂wᵢ efficiently
2. **Regularization**: Smooth volatility surface (penalize roughness)
3. **Constraints**: Enforce positivity of volatilities
4. **Market data formats**: Support standard option quotes

The calibration leverages the sparse Hessian structure for efficient second-order
optimization (e.g., L-BFGS, Trust Region).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, OptimizeResult
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aad_edge_pushing.pde.models import BSplineModel


@dataclass
class MarketDataPoint:
    """Single market quote for calibration."""
    strike: float
    implied_vol: float
    bid_vol: Optional[float] = None
    ask_vol: Optional[float] = None
    weight: float = 1.0  # Calibration weight (e.g., by vega or liquidity)


@dataclass
class CalibrationConfig:
    """Configuration for B-spline calibration."""
    # Regularization
    regularization_type: str = 'tikhonov'  # 'tikhonov', 'roughness', 'none'
    lambda_reg: float = 0.01  # Regularization strength

    # Optimization
    optimizer: str = 'L-BFGS-B'  # 'L-BFGS-B', 'SLSQP', 'trust-constr'
    max_iterations: int = 100
    tolerance: float = 1e-6

    # Constraints
    min_vol: float = 0.01  # Minimum volatility (positivity + numerical stability)
    max_vol: float = 2.0   # Maximum volatility (avoid unrealistic values)

    # Initial guess
    initial_guess_method: str = 'flat'  # 'flat', 'interpolate', 'custom'
    initial_vol: float = 0.20  # For 'flat' method

    # Logging
    verbose: bool = True


class BSplineCalibrator:
    """
    Calibrate B-spline volatility model to market implied volatilities.

    Usage:
        >>> # Create market data
        >>> strikes = np.array([80, 90, 100, 110, 120])
        >>> implied_vols = np.array([0.25, 0.22, 0.20, 0.21, 0.23])
        >>> market_data = [MarketDataPoint(K, vol) for K, vol in zip(strikes, implied_vols)]
        >>>
        >>> # Create B-spline model template
        >>> model = create_flat_bspline(n_knots=15, degree=3, volatility=0.20)
        >>>
        >>> # Calibrate
        >>> calibrator = BSplineCalibrator(model, market_data)
        >>> result = calibrator.calibrate()
        >>>
        >>> # Get calibrated model
        >>> calibrated_model = result['model']
    """

    def __init__(self,
                 model: BSplineModel,
                 market_data: List[MarketDataPoint],
                 config: Optional[CalibrationConfig] = None):
        """
        Initialize calibrator.

        Args:
            model: B-spline model template (defines knots, degree, domain)
            market_data: List of market implied volatility quotes
            config: Calibration configuration (uses defaults if None)
        """
        self.model = model
        self.market_data = market_data
        self.config = config or CalibrationConfig()

        # Extract strikes and vols
        self.strikes = np.array([d.strike for d in market_data])
        self.market_vols = np.array([d.implied_vol for d in market_data])
        self.weights = np.array([d.weight for d in market_data])

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

        # Number of parameters
        self.n_params = model.get_n_params()

        # Optimization history
        self.iteration = 0
        self.loss_history = []
        self.coeff_history = []

    def _objective_function(self, coeffs: np.ndarray) -> float:
        """
        Compute calibration loss: data fit + regularization.

        Args:
            coeffs: B-spline coefficients [w₀, ..., wₙ]

        Returns:
            Loss value
        """
        # Update model with current coefficients
        model_vols = self._evaluate_model(coeffs, self.strikes)

        # Data fitting loss (weighted MSE)
        residuals = model_vols - self.market_vols
        data_loss = np.sum(self.weights * residuals**2)

        # Regularization
        reg_loss = self._compute_regularization(coeffs)

        total_loss = data_loss + self.config.lambda_reg * reg_loss

        # Log
        self.iteration += 1
        self.loss_history.append(total_loss)
        self.coeff_history.append(coeffs.copy())

        if self.config.verbose and self.iteration % 10 == 0:
            print(f"  Iteration {self.iteration}: Loss = {total_loss:.6e} "
                  f"(Data: {data_loss:.6e}, Reg: {reg_loss:.6e})")

        return total_loss

    def _objective_gradient(self, coeffs: np.ndarray) -> np.ndarray:
        """
        Compute gradient of calibration loss wrt coefficients.

        This can be computed efficiently using:
        1. Chain rule: ∂L/∂wᵢ = Σⱼ ∂L/∂σⱼ · ∂σⱼ/∂wᵢ
        2. Sparse structure: ∂σ(S)/∂wᵢ = Bᵢ(S) (only degree+1 non-zero)

        Args:
            coeffs: B-spline coefficients

        Returns:
            Gradient vector ∂L/∂w
        """
        from scipy.interpolate import BSpline

        # Data fitting gradient
        model_vols = self._evaluate_model(coeffs, self.strikes)
        residuals = model_vols - self.market_vols

        # ∂L_data/∂wᵢ = 2 Σⱼ wⱼ·(σ_model(Kⱼ) - σ_market(Kⱼ)) · Bᵢ(Kⱼ)
        grad_data = np.zeros(self.n_params)

        for idx in range(self.n_params):
            # Evaluate basis function Bᵢ at all strikes
            basis_coeff = np.zeros(self.n_params)
            basis_coeff[idx] = 1.0
            bspline_basis = BSpline(self.model.knot_vector, basis_coeff, self.model.degree)

            basis_values = bspline_basis(self.strikes)

            # Gradient contribution from this basis function
            grad_data[idx] = 2.0 * np.sum(self.weights * residuals * basis_values)

        # Regularization gradient
        grad_reg = self._compute_regularization_gradient(coeffs)

        total_grad = grad_data + self.config.lambda_reg * grad_reg

        return total_grad

    def _evaluate_model(self, coeffs: np.ndarray, strikes: np.ndarray) -> np.ndarray:
        """Evaluate B-spline model at given strikes."""
        from scipy.interpolate import BSpline

        bspline = BSpline(self.model.knot_vector, coeffs, self.model.degree)
        return bspline(strikes)

    def _compute_regularization(self, coeffs: np.ndarray) -> float:
        """
        Compute regularization penalty.

        Options:
        - 'tikhonov': R(w) = ||w - w₀||² (smoothness from prior)
        - 'roughness': R(w) = ||D²w||² (penalize second differences)
        - 'none': R(w) = 0
        """
        if self.config.regularization_type == 'none':
            return 0.0

        elif self.config.regularization_type == 'tikhonov':
            # Penalize deviation from initial guess (typically flat volatility)
            w0 = np.full(self.n_params, self.config.initial_vol)
            return np.sum((coeffs - w0)**2)

        elif self.config.regularization_type == 'roughness':
            # Penalize roughness: sum of second differences
            # D²wᵢ = wᵢ₊₁ - 2wᵢ + wᵢ₋₁
            second_diffs = coeffs[2:] - 2*coeffs[1:-1] + coeffs[:-2]
            return np.sum(second_diffs**2)

        else:
            raise ValueError(f"Unknown regularization type: {self.config.regularization_type}")

    def _compute_regularization_gradient(self, coeffs: np.ndarray) -> np.ndarray:
        """Compute gradient of regularization term."""
        if self.config.regularization_type == 'none':
            return np.zeros(self.n_params)

        elif self.config.regularization_type == 'tikhonov':
            w0 = np.full(self.n_params, self.config.initial_vol)
            return 2.0 * (coeffs - w0)

        elif self.config.regularization_type == 'roughness':
            # Gradient of ||D²w||²
            grad = np.zeros(self.n_params)

            # Interior points: ∂R/∂wᵢ involves neighboring terms
            for i in range(self.n_params):
                if i >= 2:
                    # Term from (wᵢ - 2wᵢ₋₁ + wᵢ₋₂)
                    grad[i] += 2.0 * (coeffs[i] - 2*coeffs[i-1] + coeffs[i-2])

                if 0 < i < self.n_params - 1:
                    # Term from (wᵢ₊₁ - 2wᵢ + wᵢ₋₁)
                    grad[i] += 2.0 * (-2) * (coeffs[i+1] - 2*coeffs[i] + coeffs[i-1])

                if i < self.n_params - 2:
                    # Term from (wᵢ₊₂ - 2wᵢ₊₁ + wᵢ)
                    grad[i] += 2.0 * (coeffs[i+2] - 2*coeffs[i+1] + coeffs[i])

            return grad

        else:
            raise ValueError(f"Unknown regularization type: {self.config.regularization_type}")

    def _get_initial_guess(self) -> np.ndarray:
        """Generate initial coefficient guess."""
        if self.config.initial_guess_method == 'flat':
            # Constant volatility
            return np.full(self.n_params, self.config.initial_vol)

        elif self.config.initial_guess_method == 'interpolate':
            # Interpolate market data to knot locations
            from scipy.interpolate import interp1d

            # Simple linear interpolation of market vols to knot locations
            interp = interp1d(self.strikes, self.market_vols,
                            kind='linear', fill_value='extrapolate')

            # Evaluate at knot locations (approximately)
            # This is heuristic - proper initialization would require solving for coefficients
            interior_knots = self.model.interior_knots
            return interp(interior_knots)[:self.n_params]

        elif self.config.initial_guess_method == 'custom':
            # Use current model coefficients
            return self.model.coefficients.copy()

        else:
            raise ValueError(f"Unknown initial guess method: {self.config.initial_guess_method}")

    def calibrate(self) -> Dict:
        """
        Run calibration optimization.

        Returns:
            Dictionary with:
                - model: Calibrated BSplineModel
                - coefficients: Optimal coefficients
                - loss: Final loss value
                - n_iterations: Number of iterations
                - success: Whether optimization succeeded
                - message: Optimization status message
        """
        if self.config.verbose:
            print(f"\nCalibrating B-spline volatility model...")
            print(f"  Market data: {len(self.market_data)} quotes")
            print(f"  Parameters: {self.n_params} B-spline coefficients")
            print(f"  Strike range: [{self.strikes.min():.2f}, {self.strikes.max():.2f}]")
            print(f"  Regularization: {self.config.regularization_type} (λ={self.config.lambda_reg})")

        # Initial guess
        x0 = self._get_initial_guess()

        # Bounds (enforce positivity and maximum)
        bounds = [(self.config.min_vol, self.config.max_vol) for _ in range(self.n_params)]

        # Reset iteration counter
        self.iteration = 0
        self.loss_history = []
        self.coeff_history = []

        # Optimize
        if self.config.verbose:
            print(f"\nRunning {self.config.optimizer} optimization...")

        result: OptimizeResult = minimize(
            fun=self._objective_function,
            x0=x0,
            method=self.config.optimizer,
            jac=self._objective_gradient,
            bounds=bounds,
            options={
                'maxiter': self.config.max_iterations,
                'ftol': self.config.tolerance,
                'disp': False
            }
        )

        # Extract optimal coefficients
        optimal_coeffs = result.x

        # Create calibrated model
        calibrated_model = BSplineModel(
            knots=self.model.interior_knots,
            coefficients=optimal_coeffs,
            degree=self.model.degree,
            S_min=self.model.S_min,
            S_max=self.model.S_max
        )

        # Compute final metrics
        calibrated_vols = self._evaluate_model(optimal_coeffs, self.strikes)
        errors = calibrated_vols - self.market_vols
        rmse = np.sqrt(np.mean(errors**2))
        max_abs_error = np.max(np.abs(errors))

        if self.config.verbose:
            print(f"\nCalibration Complete:")
            print(f"  Status: {result.message}")
            print(f"  Iterations: {result.nit}")
            print(f"  Final loss: {result.fun:.6e}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Max |error|: {max_abs_error:.6f}")

        return {
            'model': calibrated_model,
            'coefficients': optimal_coeffs,
            'loss': result.fun,
            'n_iterations': result.nit,
            'success': result.success,
            'message': result.message,
            'rmse': rmse,
            'max_abs_error': max_abs_error,
            'loss_history': self.loss_history,
            'coeff_history': self.coeff_history,
            'calibrated_vols': calibrated_vols,
            'errors': errors
        }
