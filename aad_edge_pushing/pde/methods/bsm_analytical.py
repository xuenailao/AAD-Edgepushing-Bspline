"""
Method 5: BSM Analytical Solution

Black-Scholes-Merton analytical formulas.
Machine-precision Greeks (baseline for comparison).
"""

import numpy as np
from scipy.stats import norm
import time
from typing import Dict

from .base_method import HessianMethodBase


class BSMAnalyticalMethod(HessianMethodBase):
    """BSM analytical solution (baseline)."""

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float):
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "BSM-Analytical"

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        start_time = time.time()

        S = S0 if S0 is not None else self.S0
        K = self.K
        T = self.T
        r = self.r

        # BSM formula
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        phi_d1 = norm.pdf(d1)
        Phi_d1 = norm.cdf(d1)
        Phi_d2 = norm.cdf(d2)

        # Greeks
        price = S * Phi_d1 - K * np.exp(-r * T) * Phi_d2
        delta = Phi_d1
        vega = S * phi_d1 * sqrt_T
        gamma = phi_d1 / (S * sigma * sqrt_T)
        vanna = -phi_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma

        jacobian = np.array([delta, vega])
        hessian = np.array([[gamma, vanna], [vanna, volga]])

        time_ms = (time.time() - start_time) * 1000

        return self._format_result(
            price=price,
            jacobian=jacobian,
            hessian=hessian,
            time_ms=time_ms,
            n_pde_solves=0
        )
