"""
Edge-Pushing Method (V2 - Cython Optimized) for Hessian computation.

Uses the optimized symm_sparse_adjlist_v2 for improved performance.
"""

from .edge_pushing import EdgePushingMethod as EdgePushingMethodV1


class EdgePushingMethodV2(EdgePushingMethodV1):
    """
    V2 version of Edge-Pushing Method with Cython optimizations.

    Inherits all functionality from V1 but uses the optimized algo4_adjlist_v2
    internally.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method_name = "Edge-Pushing-V2"

    def _compute_hessian_internal(self, S0_val: float, sigma_val: float):
        """
        Internal Hessian computation using V2 optimized algorithm.

        Overrides parent method to use algo4_adjlist_v2 instead of algo4_adjlist.
        """
        from ...aad.core.var import ADVar
        from ...aad.core.tape import global_tape
        from ...edge_pushing.algo4_adjlist_v2 import algo4_adjlist_v2

        # Clear tape for fresh computation
        global_tape.clear()

        # Create AD variables for Hessian computation
        S0_var_h = ADVar(S0_val, name='S0')
        sigma_var_h = ADVar(sigma_val, name='sigma')

        # Solve PDE with AAD
        price_var_h = self.solver.solve_with_aad_hessian(S0_var_h, sigma_var_h)

        # Extract first-order derivatives (Jacobian)
        delta = S0_var_h.adj
        vega = sigma_var_h.adj

        # Compute Hessian using V2 optimized algorithm
        hessian = algo4_adjlist_v2(price_var_h, [S0_var_h, sigma_var_h])

        # Extract second-order derivatives
        gamma = hessian[0, 0]
        vanna = hessian[0, 1]
        volga = hessian[1, 1]

        return {
            'price': float(price_var_h.val),
            'delta': delta,
            'vega': vega,
            'gamma': gamma,
            'vanna': vanna,
            'volga': volga
        }
