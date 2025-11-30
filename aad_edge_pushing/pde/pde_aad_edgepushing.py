import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from .pde_config import PDEConfig


class BS_PDE_AAD:
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 M: int = 151, N_base: int = 50, n_rannacher: int = 4,
                 use_adaptive_Smax: bool = True, sigma_margin: float = 0.1):
        """
        BS PDE solver with AAD

        Args:
            n_rannacher: Number of Rannacher smoothing steps
            use_adaptive_Smax: Use volatility-adaptive S_max
            sigma_margin: Safety margin for bumping consistency
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N_base = N_base
        self.n_rannacher = n_rannacher

        # Adaptive S_max with margin for bumping consistency
        if use_adaptive_Smax:
            S_max = PDEConfig.compute_unified_Smax(K, T, sigma, r, sigma_margin)
        else:
            S_max = 5.0 * K

        self.S_max = S_max

        # Create log-space grid using shared config
        S_min = 1e-3
        self.x_grid, self.S_grid, self.dx = PDEConfig.create_log_space_grid(S_min, S_max, M)
        self.S0_idx = None

        self.N = N_base
        self.dt = T / self.N
        alpha = (sigma**2 / 2.0) / (self.dx**2)
        dt_critical = (self.dx**2) / (2.0 * alpha)
        self.cfl_ratio = self.dt / dt_critical

        # Note: Rannacher not implemented in AAD solver (phi always = 0.5)
        # Implementing Rannacher would require rebuilding matrices inside time loop
        # which breaks the ADVar computational graph structure
        self.phi = 0.5  # Crank-Nicolson


    def _compute_spline_second_derivatives(self, V: List[ADVar], S_grid: np.ndarray) -> List[ADVar]:
        """
        Compute second derivatives M_i for natural cubic spline

        Solves tridiagonal system with natural boundary conditions M[0] = M[-1] = 0

        Args:
            V: List of function values at grid points (ADVars)
            S_grid: Grid points (numpy array)

        Returns:
            M_vals: Second derivatives M_i at each grid point (ADVars)
        """
        n = len(V)

        if n < 3:
            # Too few points, return zeros
            return [ADVar(0.0, requires_grad=False) for _ in range(n)]

        # Grid spacings
        h = np.diff(S_grid)  # h[i] = S[i+1] - S[i]

        # Build tridiagonal system: A * M = d
        # Interior equations for i = 1, ..., n-2
        # λ_i * M_{i-1} + 2*M_i + μ_i * M_{i+1} = d_i

        # Coefficients
        lambda_vals = []  # Lower diagonal
        mu_vals = []      # Upper diagonal
        d_vals = []       # RHS

        for i in range(1, n - 1):
            h_im1 = h[i - 1]  # h_{i-1} = S_i - S_{i-1}
            h_i = h[i]        # h_i = S_{i+1} - S_i

            lambda_i = h_im1 / (h_im1 + h_i)
            mu_i = h_i / (h_im1 + h_i)

            # RHS: d_i = 6 / (h_{i-1} + h_i) * [(V_{i+1} - V_i)/h_i - (V_i - V_{i-1})/h_{i-1}]
            d_i = (ADVar(6.0) / ADVar(h_im1 + h_i)) * (
                (V[i + 1] - V[i]) / ADVar(h_i) - (V[i] - V[i - 1]) / ADVar(h_im1)
            )

            lambda_vals.append(lambda_i)
            mu_vals.append(mu_i)
            d_vals.append(d_i)

        # Solve tridiagonal system with natural boundary conditions
        # M[0] = 0, M[n-1] = 0
        n_interior = n - 2  # Number of unknowns

        if n_interior == 0:
            # Only 2 points, both boundary
            return [ADVar(0.0, requires_grad=False) for _ in range(n)]

        # Tridiagonal matrix:
        # [  2      μ_1      0    ...   0    ]
        # [ λ_2      2      μ_2   ...   0    ]
        # [  0      λ_3      2    ...   0    ]
        # [ ...     ...     ...   ...  ...   ]
        # [  0       0       0    ... λ_{n-2} 2 ]

        # Thomas algorithm for tridiagonal system
        a = [ADVar(0.0)] + [ADVar(lam) for lam in lambda_vals]  # Lower diagonal (shifted)
        b = [ADVar(2.0) for _ in range(n_interior)]  # Main diagonal
        c = [ADVar(mu) for mu in mu_vals] + [ADVar(0.0)]  # Upper diagonal (shifted)
        d = d_vals

        # Forward elimination
        c_prime = [None] * n_interior
        d_prime = [None] * n_interior

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n_interior):
            denom = b[i] - a[i] * c_prime[i - 1]
            c_prime[i] = c[i] / denom if i < n_interior - 1 else ADVar(0.0)
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

        # Back substitution
        M_interior = [None] * n_interior
        M_interior[-1] = d_prime[-1]

        for i in range(n_interior - 2, -1, -1):
            M_interior[i] = d_prime[i] - c_prime[i] * M_interior[i + 1]

        # Add boundary conditions
        M_vals = [ADVar(0.0, requires_grad=False)] + M_interior + [ADVar(0.0, requires_grad=False)]

        return M_vals

    def _terminal_condition(self) -> np.ndarray:
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_condition_left(self, t: float) -> float:
        return 0.0

    def _boundary_condition_right(self, t: float) -> float:
        T_remain = self.T - t
        return self.S_grid[-1] - self.K * np.exp(-self.r * T_remain)

    def build_tridiagonal_cn(self, sigma_var: ADVar, dt: ADVar):
        n = self.M - 2
        dx = self.dx
        dx_sq = ADVar(dx**2, requires_grad=False)
        dx_2 = ADVar(2.0 * dx, requires_grad=False)

        a_L, b_L, c_L = [], [], []
        a_R, b_R, c_R = [], [], []

        # Log-space PDE coefficients (constant for all grid points!)
        alpha = (sigma_var * sigma_var / ADVar(2.0)) / dx_sq  # Diffusion
        beta = (ADVar(self.r) - sigma_var * sigma_var / ADVar(2.0)) / dx_2  # Drift
        gamma = -ADVar(self.r)  # Discount

        l = alpha - beta  # Lower diagonal
        c = -ADVar(2.0) * alpha + gamma  # Main diagonal
        u = alpha + beta  # Upper diagonal

        for i in range(n):
            # Same coefficients for all grid points
            l_i = l
            c_i = c
            u_i = u

            phi = self.phi

            if i == 0:
                a_L.append(ADVar(0.0))
            else:
                a_L.append(-ADVar(phi) * dt * l_i)

            b_L.append(ADVar(1.0) - ADVar(phi) * dt * c_i)

            if i == n-1:
                c_L.append(ADVar(0.0))
            else:
                c_L.append(-ADVar(phi) * dt * u_i)

            if i == 0:
                a_R.append(ADVar(0.0))
            else:
                a_R.append(ADVar(1.0 - phi) * dt * l_i)

            b_R.append(ADVar(1.0) + ADVar(1.0 - phi) * dt * c_i)

            if i == n-1:
                c_R.append(ADVar(0.0))
            else:
                c_R.append(ADVar(1.0 - phi) * dt * u_i)

        return a_L, b_L, c_L, a_R, b_R, c_R

    def tridiag_solve(self, a: List[ADVar], b: List[ADVar], c: List[ADVar],
                     d: List[ADVar]) -> List[ADVar]:
        n = len(d)
        c_prime = [None] * n
        d_prime = [None] * n

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denom if i < n-1 else ADVar(0.0)
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        x = [None] * n
        x[-1] = d_prime[-1]

        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def cn_step(self, V: List[ADVar], a_L: List[ADVar], b_L: List[ADVar],
                c_L: List[ADVar], a_R: List[ADVar], b_R: List[ADVar],
                c_R: List[ADVar], t_current: float) -> List[ADVar]:
        n = self.M - 2
        rhs = [None] * n

        for i in range(n):
            if i == 0:
                V_left = ADVar(self._boundary_condition_left(t_current), requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i+1] - a_R[i] * V_left
            elif i == n-1:
                V_right = ADVar(self._boundary_condition_right(t_current), requires_grad=False)
                rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] + c_R[i] * V[i+1]

        V_new = self.tridiag_solve(a_L, b_L, c_L, rhs)
        return V_new

    def solve_pde_with_aad(self, S0_val: float, sigma_val: float,
                          compute_hessian: bool = False, verbose: bool = False):
        """
        Solve PDE with AAD - S0 and sigma both as ADVars

        Args:
            S0_val: Initial stock price
            sigma_val: Volatility
            compute_hessian: Whether to compute Hessian (Gamma, Vanna, Volga)
            verbose: Print diagnostic information
        """
        # Use optimized Cython version (4× faster than algo4_adjlist)
        # BUG FIXED: C++ SymmSparseAdjListCpp had accumulation bug, now uses Python version
        from aad_edge_pushing.edge_pushing.algo4_cython_simple import algo4_cython_simple

        t_start = time.perf_counter()
        global_tape.reset()

        # KEY CHANGE: S0 is now also an ADVar!
        S0_var = ADVar(S0_val, requires_grad=True, name="S0")
        sigma_var = ADVar(sigma_val, requires_grad=True, name="sigma")
        self.S0 = S0_val

        # Use adaptive N (self.N) instead of fixed N_base
        N = self.N
        dt_val = self.T / N
        t_grid = np.linspace(0, self.T, N + 1)

        if verbose:
            dt_vals = np.diff(t_grid)
            print(f"  Grid: M={self.M}, N={N}")
            print(f"  dt={dt_vals[0]:.6f}, dx={self.dx:.4f} (log-space)")

        dt_val = t_grid[1] - t_grid[0]
        dt = ADVar(dt_val, requires_grad=False)

        a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_cn(sigma_var, dt)

        V_terminal = self._terminal_condition()
        V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

        # Time stepping
        for n in range(N):
            t_current = t_grid[n+1]
            V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_current)

        # Step 1: Compute spline second derivatives M_i (tridiagonal solve)
        # Natural boundary conditions: M[0] = M[-1] = 0
        n_interior = len(V)  # Number of interior points
        S_interior = self.S_grid[1:-1]  # Corresponding S values

        # Build tridiagonal system for M_i
        # System: λ_i * M_{i-1} + 2*M_i + μ_i * M_{i+1} = d_i
        # where λ_i, μ_i are based on grid spacing, d_i from V values

        M_vals = self._compute_spline_second_derivatives(V, S_interior)

        # Step 2: Find interval containing S0
        idx = np.searchsorted(S_interior, S0_val)
        if idx == 0:
            idx = 1
        elif idx >= n_interior:
            idx = n_interior - 1

        # Interval [S_i, S_{i+1}] where i = idx-1
        i = idx - 1
        S_i = S_interior[i]
        S_i1 = S_interior[i + 1]
        V_i = V[i]
        V_i1 = V[i + 1]
        # M_vals has same indexing as V (both interior points)
        M_i = M_vals[i]
        M_i1 = M_vals[i + 1]

        h = S_i1 - S_i

        # Step 3: Natural cubic spline formula with S0_var (ADVar)
        # A = (S_{i+1} - s) / h,  B = (s - S_i) / h
        S_i_var = ADVar(S_i, requires_grad=False)
        S_i1_var = ADVar(S_i1, requires_grad=False)
        h_var = ADVar(h, requires_grad=False)

        A = (S_i1_var - S0_var) / h_var
        B = (S0_var - S_i_var) / h_var

        # Cubic terms
        A3 = A * A * A
        B3 = B * B * B

        # Natural spline interpolation formula
        # p(s) = A*V_i + B*V_{i+1} + [(A³-A)*h²/6]*M_i + [(B³-B)*h²/6]*M_{i+1}
        h2_over_6 = h_var * h_var / ADVar(6.0)

        price_var = (A * V_i + B * V_i1 +
                    (A3 - A) * h2_over_6 * M_i +
                    (B3 - B) * h2_over_6 * M_i1)

        # Store interval info for Hessian computation
        spline_info = {
            'i': i,
            'S_i': S_i,
            'S_i1': S_i1,
            'V_i': V_i,
            'V_i1': V_i1,
            'M_i': M_i,
            'M_i1': M_i1,
            'h': h,
            'M_vals': M_vals
        }

        price = price_var.val

        # Jacobian via backward pass
        price_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)

        delta = S0_var.adj  # ∂V/∂S0 via AAD!
        vega = sigma_var.adj

        t_end = time.perf_counter()
        time_ms = (t_end - t_start) * 1000.0

        result = {
            'price': price,
            'delta': delta,
            'vega': vega,
            'time_ms': time_ms,
            'jacobian': np.array([delta, vega])
        }

        # Hessian via Edge-Pushing
        if compute_hessian:
            global_tape.reset()

            # Recompute with fresh tape
            S0_var_h = ADVar(S0_val, requires_grad=True, name="S0")
            sigma_var_h = ADVar(sigma_val, requires_grad=True, name="sigma")

            dt_h = ADVar(dt_val, requires_grad=False)
            a_L_h, b_L_h, c_L_h, a_R_h, b_R_h, c_R_h = self.build_tridiagonal_cn(sigma_var_h, dt_h)

            V_h = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]
            for n in range(N):
                t_current = t_grid[n+1]
                V_h = self.cn_step(V_h, a_L_h, b_L_h, c_L_h, a_R_h, b_R_h, c_R_h, t_current)

            # Natural cubic spline interpolation with S0_var_h
            # Recompute spline second derivatives
            M_vals_h = self._compute_spline_second_derivatives(V_h, S_interior)

            # Use same interval from Jacobian
            i = spline_info['i']
            S_i = spline_info['S_i']
            S_i1 = spline_info['S_i1']
            h = spline_info['h']
            V_i_h = V_h[i]
            V_i1_h = V_h[i + 1]
            # M_vals_h has same indexing as V_h (both interior points)
            M_i_h = M_vals_h[i]
            M_i1_h = M_vals_h[i + 1]

            # Spline formula with S0_var_h
            S_i_var_h = ADVar(S_i, requires_grad=False)
            S_i1_var_h = ADVar(S_i1, requires_grad=False)
            h_var_h = ADVar(h, requires_grad=False)

            A_h = (S_i1_var_h - S0_var_h) / h_var_h
            B_h = (S0_var_h - S_i_var_h) / h_var_h

            A3_h = A_h * A_h * A_h
            B3_h = B_h * B_h * B_h

            h2_over_6_h = h_var_h * h_var_h / ADVar(6.0)

            price_var_h = (A_h * V_i_h + B_h * V_i1_h +
                          (A3_h - A_h) * h2_over_6_h * M_i_h +
                          (B3_h - B_h) * h2_over_6_h * M_i1_h)

            # Edge-Pushing for full 2×2 Hessian (using optimized Cython version)
            hessian = algo4_cython_simple(price_var_h, [S0_var_h, sigma_var_h])

            # Extract all second-order Greeks
            gamma = hessian[0, 0]
            vanna = hessian[0, 1]
            volga = hessian[1, 1]

            result['gamma'] = gamma
            result['vanna'] = vanna
            result['volga'] = volga
            result['hessian'] = hessian

        return result
