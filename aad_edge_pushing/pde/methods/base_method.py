"""
Base abstract class for Hessian computation methods.

All methods compute the 2x2 Hessian matrix with respect to (S0, sigma):

    H = [[∂²V/∂S0²,  ∂²V/∂S0∂σ],     [[gamma, vanna],
         [∂²V/∂S0∂σ, ∂²V/∂σ²  ]]  =    [vanna, volga]]

Standard interface ensures consistent testing and comparison.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict
import time


class HessianMethodBase(ABC):
    """
    Abstract base class for all Hessian computation methods.

    Attributes:
        M (int): Number of spatial grid points
        N (int): Number of time steps
        method_name (str): Name of the method
        S0 (float): Initial stock price
        K (float): Strike price
        T (float): Time to maturity
        r (float): Risk-free rate
    """

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float):
        """
        Initialize method with PDE parameters.

        Args:
            M: Number of spatial grid points
            N: Number of time steps
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
        """
        self.M = M
        self.N = N
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.method_name = "Base"

    @abstractmethod
    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        """
        Compute option price, Jacobian, and Hessian matrix.

        Args:
            S0: Initial stock price (can override constructor value)
            sigma: Volatility

        Returns:
            Dictionary with standard format:
            {
                'price': float,                          # Option price V(S0, σ)
                'jacobian': np.array([delta, vega]),     # [∂V/∂S0, ∂V/∂σ]
                'hessian': np.array([[gamma, vanna],     # [[∂²V/∂S0², ∂²V/∂S0∂σ],
                                     [vanna, volga]]),   #  [∂²V/∂S0∂σ, ∂²V/∂σ²]]
                'greeks': {                              # Individual Greeks for convenience
                    'delta': float,
                    'gamma': float,
                    'vega': float,
                    'vanna': float,
                    'volga': float
                },
                'time_ms': float,                        # Computation time in milliseconds
                'n_pde_solves': int,                     # Number of PDE solves performed
                'method': str                            # Method name
            }
        """
        pass

    def _format_result(self, price: float, jacobian: np.ndarray,
                      hessian: np.ndarray, time_ms: float,
                      n_pde_solves: int) -> Dict:
        """
        Format results into standard output dictionary.

        Args:
            price: Option price
            jacobian: [delta, vega]
            hessian: [[gamma, vanna], [vanna, volga]]
            time_ms: Computation time
            n_pde_solves: Number of PDE solves

        Returns:
            Standardized result dictionary
        """
        delta, vega = jacobian[0], jacobian[1]
        gamma, vanna = hessian[0, 0], hessian[0, 1]
        volga = hessian[1, 1]

        return {
            'price': price,
            'jacobian': jacobian,
            'hessian': hessian,
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'vanna': vanna,
                'volga': volga
            },
            'time_ms': time_ms,
            'n_pde_solves': n_pde_solves,
            'method': self.method_name
        }

    def __repr__(self):
        return f"{self.method_name}(M={self.M}, N={self.N})"
