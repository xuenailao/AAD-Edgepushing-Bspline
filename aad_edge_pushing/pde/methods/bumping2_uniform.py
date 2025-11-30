"""
Method 1: Double Bumping (Bumping2) - Uniform Grid Version

使用均匀空间网格（非对数网格）的有限差分方法
"""

import numpy as np
import time
from typing import Dict
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from .base_method import HessianMethodBase


class Bumping2UniformMethod(HessianMethodBase):
    """
    使用均匀网格的纯有限差分方法
    """

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float,
                 eps_S: float = None, eps_sigma: float = 0.01,
                 n_rannacher: int = 4, use_adaptive_Smax: bool = True, sigma_margin: float = 0.1):
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "Bumping2_Uniform"
        self.eps_S = eps_S if eps_S is not None else max(2.0, 0.005 * S0)
        self.eps_sigma = eps_sigma
        self.n_rannacher = n_rannacher
        self.use_adaptive_Smax = use_adaptive_Smax
        self.sigma_margin = sigma_margin

    def _solve_pde(self, S0: float, sigma: float, S_max_override: float = None) -> float:
        """使用均匀网格求解PDE"""
        from aad_edge_pushing.pde.uniform_pde_solver import UniformPDESolver

        solver = UniformPDESolver(S0=S0, K=self.K, T=self.T, r=self.r, sigma=sigma,
                                M=self.M, N_base=self.N,
                                n_rannacher=self.n_rannacher,
                                use_adaptive_Smax=self.use_adaptive_Smax,
                                sigma_margin=self.sigma_margin,
                                S_max_override=S_max_override)
        price, _ = solver._solve_pde_numerical(S0, sigma)
        return price

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        start_time = time.time()

        S = S0 if S0 is not None else self.S0
        eps_S = self.eps_S
        eps_sigma = self.eps_sigma

        # Compute unified S_max
        if self.use_adaptive_Smax:
            from aad_edge_pushing.pde.pde_config import PDEConfig
            S_max_unified = PDEConfig.compute_unified_Smax(
                self.K, self.T, sigma, self.r, sigma_margin=eps_sigma
            )
        else:
            S_max_unified = None

        # 1. Base value
        V0 = self._solve_pde(S, sigma, S_max_unified)

        # 2. Perturb S0
        V_Sp = self._solve_pde(S + eps_S, sigma, S_max_unified)
        V_Sm = self._solve_pde(S - eps_S, sigma, S_max_unified)

        # 3. Perturb sigma
        V_sp = self._solve_pde(S, sigma + eps_sigma, S_max_unified)
        V_sm = self._solve_pde(S, sigma - eps_sigma, S_max_unified)

        # 4. Cross perturbations for Vanna
        V_Sp_sp = self._solve_pde(S + eps_S, sigma + eps_sigma, S_max_unified)
        V_Sm_sp = self._solve_pde(S - eps_S, sigma + eps_sigma, S_max_unified)
        V_Sp_sm = self._solve_pde(S + eps_S, sigma - eps_sigma, S_max_unified)
        V_Sm_sm = self._solve_pde(S - eps_S, sigma - eps_sigma, S_max_unified)

        # === Compute Jacobian ===
        delta = (V_Sp - V_Sm) / (2 * eps_S)
        vega = (V_sp - V_sm) / (2 * eps_sigma)

        # === Compute Hessian ===
        gamma = (V_Sp - 2*V0 + V_Sm) / (eps_S**2)
        volga = (V_sp - 2*V0 + V_sm) / (eps_sigma**2)

        delta_sp = (V_Sp_sp - V_Sm_sp) / (2 * eps_S)
        delta_sm = (V_Sp_sm - V_Sm_sm) / (2 * eps_S)
        vanna = (delta_sp - delta_sm) / (2 * eps_sigma)

        jacobian = np.array([delta, vega])
        hessian = np.array([[gamma, vanna], [vanna, volga]])

        time_ms = (time.time() - start_time) * 1000

        return self._format_result(
            price=V0,
            jacobian=jacobian,
            hessian=hessian,
            time_ms=time_ms,
            n_pde_solves=9
        )
