"""
Method 3b: True Double AAD using Forward-over-Reverse (FoR)

This implements the "pure" double-AAD method using hvp_for (Hessian-vector product).
- Does NOT use algo4_adjlist (Edge-Pushing optimization)
- Uses N calls to hvp_for to compute N columns of Hessian
- Each hvp_for call does: 1 forward pass + 1 enhanced reverse pass

For N=2 inputs (S0, sigma):
- Call 1: hvp_for with v=[1,0] -> Hessian column 1
- Call 2: hvp_for with v=[0,1] -> Hessian column 2

This is the canonical "double-AAD" method:
1. First AAD: Forward pass builds computation graph
2. Second AAD: Enhanced reverse pass propagates (adj, adj_dot) for Hessian

Cost: N × (forward + enhanced_reverse) = 2 × PDE_solve for N=2
"""

import numpy as np
import time
from typing import Dict
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from .base_method import HessianMethodBase


class DoubleAADMethod(HessianMethodBase):
    """
    True Double-AAD using Forward-over-Reverse (FoR) via hvp_for.

    This is the "pure" AAD method without Edge-Pushing optimizations.
    """

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float):
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "Double-AAD"

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        start_time = time.time()

        S = S0 if S0 is not None else self.S0

        from aad_edge_pushing.aad.core.engine import hvp_for
        from aad_edge_pushing.aad.core.tape import global_tape
        from aad_edge_pushing.aad.core.var import ADVar
        from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

        # First, get price and Jacobian (needed for output)
        solver = BS_PDE_AAD(S0=S, K=self.K, T=self.T, r=self.r, sigma=sigma,
                           M=self.M, N_base=self.N)
        result_jac = solver.solve_pde_with_aad(S, sigma, compute_hessian=False, verbose=False)
        price = result_jac['price']
        delta = result_jac['delta']
        vega = result_jac['vega']

        # Define PDE solver as a function compatible with hvp_for
        def pde_price_func(input_vars_dict):
            """
            Wrapper function that solves PDE given S0 and sigma as ADVars.
            This will be called by hvp_for for each Hessian column.

            Returns: price as ADVar
            """
            S0_var = input_vars_dict['S0']
            sigma_var = input_vars_dict['sigma']

            # Reset tape for clean computation graph
            global_tape.reset()

            # Create solver instance (sigma value from closure)
            pde_solver = BS_PDE_AAD(S0=self.S0, K=self.K, T=self.T, r=self.r, sigma=sigma,
                                    M=self.M, N_base=self.N)

            # Build PDE computation graph (replicate the logic from solve_pde_with_aad)
            dt_val = pde_solver.T / pde_solver.N_base
            dt = ADVar(dt_val, requires_grad=False)

            # Build tridiagonal system
            a_L, b_L, c_L, a_R, b_R, c_R = pde_solver.build_tridiagonal_cn(sigma_var, dt)

            # Terminal condition
            V_terminal = pde_solver._terminal_condition()
            V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

            # Time stepping
            N_steps = pde_solver.N_base
            t_grid = np.linspace(0, pde_solver.T, N_steps + 1)

            for n in range(N_steps):
                t_current = t_grid[n + 1]
                V = pde_solver.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_current)

            # Spline interpolation to S0
            S_interior = pde_solver.S_grid[1:-1]
            M_vals = pde_solver._compute_spline_second_derivatives(V, S_interior)

            # Find interpolation bracket
            S0_val_float = float(S0_var.val)
            idx = np.searchsorted(S_interior, S0_val_float)
            idx = max(0, min(idx, len(S_interior) - 2))

            i = idx
            S_i = S_interior[i]
            S_i1 = S_interior[i + 1]
            h = S_i1 - S_i

            V_i = V[i]
            V_i1 = V[i + 1]
            M_i = M_vals[i]
            M_i1 = M_vals[i + 1]

            # Cubic spline evaluation
            S_i_var = ADVar(S_i, requires_grad=False)
            S_i1_var = ADVar(S_i1, requires_grad=False)
            h_var = ADVar(h, requires_grad=False)

            A = (S_i1_var - S0_var) / h_var
            B = (S0_var - S_i_var) / h_var
            A3 = A * A * A
            B3 = B * B * B
            h2_over_6 = h_var * h_var / ADVar(6.0)

            price_var = (A * V_i + B * V_i1 +
                        (A3 - A) * h2_over_6 * M_i +
                        (B3 - B) * h2_over_6 * M_i1)

            return price_var

        # Compute Hessian using hvp_for loop
        input_keys = ['S0', 'sigma']
        inputs = {'S0': S, 'sigma': sigma}
        N_vars = len(input_keys)
        hessian = np.zeros((N_vars, N_vars))

        for j in range(N_vars):
            # Create unit vector e_j
            v_direction = {k: 0.0 for k in input_keys}
            v_direction[input_keys[j]] = 1.0

            # Compute H·e_j (jth column of Hessian)
            hv_column = hvp_for(pde_price_func, inputs, v_direction)
            hessian[:, j] = hv_column

        # Extract Greeks from Hessian
        gamma = hessian[0, 0]  # ∂²V/∂S0²
        vanna = hessian[0, 1]  # ∂²V/∂S0∂σ
        volga = hessian[1, 1]  # ∂²V/∂σ²

        jacobian = np.array([delta, vega])

        time_ms = (time.time() - start_time) * 1000

        return self._format_result(
            price=price,
            jacobian=jacobian,
            hessian=hessian,
            time_ms=time_ms,
            n_pde_solves=3  # 1 for Jacobian + 2 for Hessian (2 hvp_for calls)
        )
