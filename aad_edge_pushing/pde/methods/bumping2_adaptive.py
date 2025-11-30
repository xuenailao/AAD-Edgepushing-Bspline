"""
Method 1: Double Bumping (Bumping2) - Adaptive Grid Version

根据参数自动选择最佳网格类型：
- 低波动率(σ<0.2)或短期(T<0.5): 均匀网格
- 其他情况: 对数网格

提供最佳精度和稳健性
"""

import numpy as np
import time
from typing import Dict
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from .base_method import HessianMethodBase


class Bumping2AdaptiveMethod(HessianMethodBase):
    """
    自适应网格选择的有限差分方法
    根据σ和T自动选择对数网格或均匀网格
    """

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float,
                 eps_S: float = None, eps_sigma: float = 0.01,
                 n_rannacher: int = 4, use_adaptive_Smax: bool = True,
                 sigma_margin: float = 0.1,
                 sigma_threshold: float = 0.2,
                 T_threshold: float = 0.5,
                 auto_M: bool = False):
        """
        Initialize adaptive grid Bumping2 solver

        Args:
            M: Number of spatial grid points (base value)
            N: Number of time steps (base value)
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            eps_S: Bump size for S (if None, auto-computed)
            eps_sigma: Bump size for sigma
            n_rannacher: Number of Rannacher smoothing steps
            use_adaptive_Smax: If True, S_max adapts to sigma
            sigma_margin: Safety margin for bumping
            sigma_threshold: Threshold for grid type selection (default 0.2)
            T_threshold: Threshold for grid type selection (default 0.5)
            auto_M: If True, automatically adjust M based on sigma
        """
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "Bumping2_Adaptive"
        self.eps_S = eps_S if eps_S is not None else max(2.0, 0.005 * S0)
        self.eps_sigma = eps_sigma
        self.n_rannacher = n_rannacher
        self.use_adaptive_Smax = use_adaptive_Smax
        self.sigma_margin = sigma_margin
        self.sigma_threshold = sigma_threshold
        self.T_threshold = T_threshold
        self.auto_M = auto_M

        # Store base M, N for adaptive adjustment
        self.M_base = M
        self.N_base = N

    def _select_grid_type(self, sigma: float, T: float) -> str:
        """
        自动选择网格类型 - 基于实验数据优化的决策规则

        优化规则 (基于详细测试数据):
        1. σ < 0.18: 均匀网格 (低σ优势4-5倍)
        2. σ ≥ 0.5: 对数网格 (高σ必须，均匀网格失效)
        3. 0.18 ≤ σ < 0.5 (过渡区间):
           a. σ < 0.25: 均匀网格
           b. 0.25 ≤ σ < 0.35 (陷阱区间):
              - 短期(T<1.0): 均匀网格
              - 长期(T≥1.0): 对数网格
           c. σ ≥ 0.35: 对数网格

        Args:
            sigma: Volatility
            T: Time to maturity

        Returns:
            'uniform' or 'log'
        """
        # 规则1: 低波动率 -> 均匀网格 (优势明显)
        if sigma < 0.18:
            return 'uniform'

        # 规则2: 高波动率 -> 对数网格 (必须用对数)
        if sigma >= 0.5:
            return 'log'

        # 规则3: 过渡区间 0.18 ≤ σ < 0.5
        if sigma < 0.25:
            # 0.18 ≤ σ < 0.25: 均匀网格通常更好
            return 'uniform'
        elif sigma < 0.35:
            # 0.25 ≤ σ < 0.35: 陷阱区间，根据T选择
            # 短期用均匀，长期用对数
            return 'uniform' if T < 1.0 else 'log'
        else:
            # 0.35 ≤ σ < 0.5: 对数网格
            return 'log'

    def _adjust_M(self, sigma: float, T: float, grid_type: str) -> int:
        """
        根据参数自适应调整M

        理论依据:
        - Gamma峰宽 ≈ σ√T
        - 需要至少15-20个网格点覆盖峰值区域

        Args:
            sigma: Volatility
            T: Time to maturity
            grid_type: 'uniform' or 'log'

        Returns:
            Adjusted M value
        """
        if not self.auto_M:
            return self.M_base

        # 估计Gamma峰宽度（相对于K）
        peak_width_ratio = sigma * np.sqrt(T)

        # 根据峰宽度计算需要的M
        if grid_type == 'uniform':
            # 均匀网格：需要更多点
            # 目标：峰值区域(K±2σ√T*K)至少20个点
            points_in_peak = 20
            M_required = int(points_in_peak / (4 * peak_width_ratio))
        else:
            # 对数网格：自然在ATM附近密集
            # 较小的M即可
            points_in_peak = 15
            M_required = int(points_in_peak / (3 * peak_width_ratio))

        # 限制M范围
        M_min = 100
        M_max = 800
        M_adjusted = max(M_min, min(M_max, M_required))

        # 倾向于使用base M或更大
        M_final = max(self.M_base, M_adjusted)

        return M_final

    def _solve_pde(self, S0: float, sigma: float, S_max_override: float = None,
                   grid_type: str = None) -> float:
        """
        使用自适应网格求解PDE

        Args:
            S0: Stock price
            sigma: Volatility
            S_max_override: Override S_max
            grid_type: Force grid type ('uniform' or 'log'), or None for auto

        Returns:
            Option price at S0
        """
        # 自动选择网格类型
        if grid_type is None:
            grid_type = self._select_grid_type(sigma, self.T)

        # 自适应调整M
        M_used = self._adjust_M(sigma, self.T, grid_type)

        if grid_type == 'uniform':
            from aad_edge_pushing.pde.uniform_pde_solver import UniformPDESolver
            solver = UniformPDESolver(
                S0=S0, K=self.K, T=self.T, r=self.r, sigma=sigma,
                M=M_used, N_base=self.N,
                n_rannacher=self.n_rannacher,
                use_adaptive_Smax=self.use_adaptive_Smax,
                sigma_margin=self.sigma_margin,
                S_max_override=S_max_override
            )
        else:  # 'log'
            from aad_edge_pushing.pde.simple_pde_solver import SimplePDESolver
            solver = SimplePDESolver(
                S0=S0, K=self.K, T=self.T, r=self.r, sigma=sigma,
                M=M_used, N_base=self.N,
                n_rannacher=self.n_rannacher,
                use_adaptive_Smax=self.use_adaptive_Smax,
                sigma_margin=self.sigma_margin,
                S_max_override=S_max_override
            )

        price, _ = solver._solve_pde_numerical(S0, sigma)
        return price

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        """
        计算Hessian矩阵（二阶Greeks）

        使用自适应网格选择策略:
        - 低σ或短期T: 均匀网格
        - 其他: 对数网格

        Returns:
            包含price, jacobian, hessian, greeks的字典
            额外信息: grid_type_used, M_used
        """
        start_time = time.time()

        S = S0 if S0 is not None else self.S0
        eps_S = self.eps_S
        eps_sigma = self.eps_sigma

        # 选择网格类型
        grid_type = self._select_grid_type(sigma, self.T)
        M_used = self._adjust_M(sigma, self.T, grid_type)

        # Compute unified S_max
        if self.use_adaptive_Smax:
            from aad_edge_pushing.pde.pde_config import PDEConfig
            S_max_unified = PDEConfig.compute_unified_Smax(
                self.K, self.T, sigma, self.r, sigma_margin=eps_sigma
            )
        else:
            S_max_unified = None

        # 1. Base value
        V0 = self._solve_pde(S, sigma, S_max_unified, grid_type)

        # 2. Perturb S0
        V_Sp = self._solve_pde(S + eps_S, sigma, S_max_unified, grid_type)
        V_Sm = self._solve_pde(S - eps_S, sigma, S_max_unified, grid_type)

        # 3. Perturb sigma
        V_sp = self._solve_pde(S, sigma + eps_sigma, S_max_unified, grid_type)
        V_sm = self._solve_pde(S, sigma - eps_sigma, S_max_unified, grid_type)

        # 4. Cross perturbations for Vanna
        V_Sp_sp = self._solve_pde(S + eps_S, sigma + eps_sigma, S_max_unified, grid_type)
        V_Sm_sp = self._solve_pde(S - eps_S, sigma + eps_sigma, S_max_unified, grid_type)
        V_Sp_sm = self._solve_pde(S + eps_S, sigma - eps_sigma, S_max_unified, grid_type)
        V_Sm_sm = self._solve_pde(S - eps_S, sigma - eps_sigma, S_max_unified, grid_type)

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

        result = self._format_result(
            price=V0,
            jacobian=jacobian,
            hessian=hessian,
            time_ms=time_ms,
            n_pde_solves=9
        )

        # Add adaptive grid info
        result['grid_type_used'] = grid_type
        result['M_used'] = M_used
        result['M_base'] = self.M_base

        return result
