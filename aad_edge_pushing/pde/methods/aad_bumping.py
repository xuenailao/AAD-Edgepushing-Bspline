"""
Method 2: AAD + Bumping (Hybrid Method)

Hybrid approach: Jacobian via AAD, Hessian via bumping on Jacobian.

Strategy:
    1. Compute Jacobian at (S0, σ) via AAD (1 PDE solve)
    2. Bump S0: Compute Jacobian at (S0±ε, σ) via AAD (2 PDE solves)
    3. Bump σ: Compute Jacobian at (S0, σ±ε) via AAD (2 PDE solves)
    4. Finite difference the Jacobians to get Hessian

PDE Solves: 5 total
"""

import numpy as np
import time
from typing import Dict
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from .base_method import HessianMethodBase


class AADBumpingMethod(HessianMethodBase):
    """Hybrid AAD + Bumping method."""

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float,
                 eps_S: float = 1.0, eps_sigma: float = 0.01):
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "AAD+Bumping"
        self.eps_S = eps_S
        self.eps_sigma = eps_sigma

    def _compute_jacobian_aad(self, S0: float, sigma: float) -> tuple:
        """Compute Jacobian using AAD"""
        from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

        solver = BS_PDE_AAD(S0=S0, K=self.K, T=self.T, r=self.r, sigma=sigma,
                           M=self.M, N_base=self.N)
        result = solver.solve_pde_with_aad(
            S0_val=S0,
            sigma_val=sigma,
            compute_hessian=False,
            verbose=False
        )
        return result['price'], result['delta'], result['vega']

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        start_time = time.time()

        S = S0 if S0 is not None else self.S0
        eps_S = self.eps_S
        eps_sigma = self.eps_sigma

        # 1. Base Jacobian
        price, delta_0, vega_0 = self._compute_jacobian_aad(S, sigma)

        # 2. Perturb S0
        _, delta_Sp, _ = self._compute_jacobian_aad(S + eps_S, sigma)
        _, delta_Sm, _ = self._compute_jacobian_aad(S - eps_S, sigma)

        # 3. Perturb sigma
        _, delta_sp, vega_sp = self._compute_jacobian_aad(S, sigma + eps_sigma)
        _, delta_sm, vega_sm = self._compute_jacobian_aad(S, sigma - eps_sigma)

        # === Jacobian ===
        delta = delta_0
        vega = vega_0

        # === Hessian ===
        gamma = (delta_Sp - delta_Sm) / (2 * eps_S)
        volga = (vega_sp - vega_sm) / (2 * eps_sigma)
        vanna = (delta_sp - delta_sm) / (2 * eps_sigma)

        jacobian = np.array([delta, vega])
        hessian = np.array([[gamma, vanna], [vanna, volga]])

        time_ms = (time.time() - start_time) * 1000

        return self._format_result(
            price=price,
            jacobian=jacobian,
            hessian=hessian,
            time_ms=time_ms,
            n_pde_solves=5
        )
