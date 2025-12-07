import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import time
from aad_edge_pushing.aad.ops.transcendental import exp

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape


class FlatSVI_PDE_AAD:
    def __init__(self, S0: float, K: float, T: float, r: float,
                 T_slices: List[float], w_slices: List[float],
                 M: int = 151, N_base: int = 150):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.T_slices = T_slices
        # Convert w_slices to ADVar (requires_grad=True), unique names
        self.w_slices = [ADVar(w, requires_grad=True, name=f"w{i}") for i, w in enumerate(w_slices)]
        self.M = M
        self.N_base = N_base
        self.phi = 0.5

        # Adaptive S_max that scales with volatility
        # Expanded from 3σ to 5σ to capture full option tail at high volatility
        # Use max implied vol for scaling
        # (use float values for grid sizing)
        max_sigma = np.sqrt(max([float(w.val) if isinstance(w, ADVar) else float(w) for w in self.w_slices]) / max(T_slices))
        S_max = max(5.0 * K, S0 * np.exp((r + 5*max_sigma) * T))

        # Log-scale grid: x = log(S)
        # Transforms PDE to constant diffusion coefficient 0.5*σ²
        S_min = 1e-3
        x_min = np.log(S_min)
        x_max = np.log(S_max)

        # Uniform grid in log-space
        self.x_grid = np.linspace(x_min, x_max, M)
        self.dx = self.x_grid[1] - self.x_grid[0]

        # Convert to S-space
        self.S_grid = np.exp(self.x_grid)
        self.S0_idx = None


        self.N = N_base
        self.dt = T / self.N
        alpha = (max_sigma**2 / 2.0) / (self.dx**2)
        dt_critical = (self.dx**2) / (2.0 * alpha)
        self.cfl_ratio = self.dt / dt_critical


    def sigma_loc(self, t: float):
        # All arithmetic must use ADVar-safe ops!
        # w_slices is now a list of ADVar
        if t <= self.T_slices[0]:
            dw_dt = (self.w_slices[1] - self.w_slices[0]) / (self.T_slices[1] - self.T_slices[0])
        elif t >= self.T_slices[-1]:
            dw_dt = (self.w_slices[-1] - self.w_slices[-2]) / (self.T_slices[-1] - self.T_slices[-2])
        else:
            for i in range(len(self.T_slices) - 1):
                if self.T_slices[i] <= t <= self.T_slices[i + 1]:
                    dw_dt = (self.w_slices[i + 1] - self.w_slices[i]) / (self.T_slices[i + 1] - self.T_slices[i])
                    break
        # Ensure differentiability: use ADVar.max and .sqrt()
        # If dw_dt is ADVar, use dw_dt.max(1e-12) if available, else do manual
        # Instead, for ADVar: (dw_dt if dw_dt > 1e-12 else ADVar(1e-12))
        # But to ensure always differentiable: dw_dt + (1e-12) if dw_dt < 1e-12
        # But better: dw_dt = dw_dt if dw_dt.val > 1e-12 else ADVar(1e-12)
        # But for robustness, use: dw_dt = dw_dt + (1e-12) if dw_dt.val < 1e-12 else dw_dt
        # Instead, use: dw_dt = dw_dt if float(dw_dt.val) > 1e-12 else ADVar(1e-12)
        # But to avoid branch, use: dw_dt = dw_dt + ADVar(1e-12)
        # But this would bias the value. So, use a "softplus" or similar.
        # To keep it simple, use: (dw_dt if float(dw_dt.val) > 1e-12 else ADVar(1e-12))
        if isinstance(dw_dt, ADVar):
            if float(dw_dt.val) > 1e-12:
                return dw_dt ** 0.5
            else:
                return ADVar(1e-12, requires_grad=False) ** 0.5
        else:
            return np.sqrt(max(dw_dt, 1e-12))

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

    from aad_edge_pushing.aad.ops.transcendental import exp

    def _terminal_condition(self, K_var: ADVar, eps: float = 0.05) -> list[ADVar]:
        """Smooth approximation to max(S - K, 0) that keeps K_var in the AD graph.
        Uses a bounded sigmoid to avoid exp overflow.
        """
        payoffs: list[ADVar] = []
        one = ADVar(1.0, requires_grad=False)
        eps_var = ADVar(eps, requires_grad=False)
        for S in self.S_grid:
            S_var = ADVar(S, requires_grad=False)
            z = (S_var - K_var) / eps_var  # may be ADVar
            # clamp z to avoid overflow in exp
            z_val = float(z.val)
            if z_val > 30.0:
                sig = one
            elif z_val < -30.0:
                sig = ADVar(0.0, requires_grad=False)
            else:
                sig = one / (one + exp(-z))
            payoff = (S_var - K_var) * sig
            payoffs.append(payoff)
        return payoffs

    def _boundary_condition_left(self, t: float) -> float:
        return 0.0

    def _boundary_condition_right(self, t: float, r_var: ADVar, K_var: ADVar = None, T_var: ADVar = None) -> ADVar:
        # Use T_var if provided, else fallback to self.T
        if T_var is not None:
            if isinstance(t, ADVar):
                T_remain = T_var - t
            else:
                T_remain = T_var - ADVar(t, requires_grad=False)
        else:
            T_remain = ADVar(self.T - float(t.val) if isinstance(t, ADVar) else self.T - t, requires_grad=False)
        # Use K_var if provided, else fallback to self.K
        if K_var is None:
            K_used = ADVar(self.K, requires_grad=False)
        else:
            K_used = K_var
        return ADVar(self.S_grid[-1], requires_grad=False) - K_used * exp(-r_var * T_remain)

    def build_tridiagonal_cn(self, sigma_var: ADVar, dt: ADVar, r_var: ADVar, K_var: ADVar):
        n = self.M - 2
        dx = self.dx
        dx_sq = ADVar(dx**2, requires_grad=False)
        dx_2 = ADVar(2.0 * dx, requires_grad=False)

        a_L, b_L, c_L = [], [], []
        a_R, b_R, c_R = [], [], []

        # Log-space PDE coefficients (constant for all grid points!)
        # Add K_factor scaling
        K_factor = K_var / ADVar(self.K, requires_grad=False)
        alpha = ((sigma_var * sigma_var / ADVar(2.0)) / dx_sq) * K_factor  # Diffusion
        beta = ((r_var - sigma_var * sigma_var / ADVar(2.0)) / dx_2) * K_factor
        gamma = -r_var * K_factor

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
                c_R: List[ADVar], t_current, r_var: ADVar, K_var: ADVar = None, T_var: ADVar = None) -> List[ADVar]:
        n = self.M - 2
        rhs = [None] * n

        for i in range(n):
            if i == 0:
                V_left = ADVar(self._boundary_condition_left(t_current), requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i+1] - a_R[i] * V_left
            elif i == n-1:
                V_right = self._boundary_condition_right(t_current, r_var, K_var, T_var)
                rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] + c_R[i] * V[i+1]

        V_new = self.tridiag_solve(a_L, b_L, c_L, rhs)
        return V_new

    def solve_pde_with_aad(self, S0_val: float,
                          compute_hessian: bool = False, verbose: bool = False):
        """
        Solve PDE with AAD - S0 as ADVar and local volatility from flat SVI slices

        Args:
            S0_val: Initial stock price
            compute_hessian: Whether to compute Hessian (Gamma, Vanna, Volga)
            verbose: Print diagnostic information
        """
        from aad_edge_pushing.edge_pushing.algo4_adjlist import algo4_adjlist

        t_start = time.perf_counter()
        global_tape.reset()

        # KEY CHANGE: S0 is now also an ADVar!
        S0_var = ADVar(S0_val, requires_grad=True, name="S0")
        K_var = ADVar(self.K, requires_grad=True, name="K")
        r_var = ADVar(self.r, requires_grad=True, name="r")
        T_var = ADVar(self.T, requires_grad=True, name="T")
        N = self.N
        dt = T_var / ADVar(N, requires_grad=False)
        self.S0 = S0_val

        # Build t_grid_ad using ADVar so that T_var participates in AAD graph
        t_grid_ad = [ADVar(0.0, requires_grad=False)]
        for i in range(1, N + 1):
            t_grid_ad.append(ADVar(i, requires_grad=False) * dt)

        if verbose:
            dt_val = float(dt.val)
            print(f"  Grid: M={self.M}, N={N}")
            print(f"  dt={dt_val:.6f}, dx={self.dx:.4f} (log-space)")

        V_terminal = self._terminal_condition(K_var)
        V = V_terminal[1:-1]

        # Time stepping
        for n in range(N):
            t_current_ad = t_grid_ad[n + 1]
            sigma_local = self.sigma_loc(float(t_current_ad.val))
            a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_cn(sigma_local, dt, r_var, K_var)
            V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_current_ad, r_var, K_var, T_var)

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

        # Sensitivities to w_slices
        w_sensitivities = np.array([w.adj for w in self.w_slices])

        result = {
            'price': price,
            'delta': delta,
            'jacobian': np.array([delta])
        }
        result['w_sensitivities'] = w_sensitivities

        # Hessian via Edge-Pushing
        if compute_hessian:
            global_tape.reset()
            # Recompute with fresh tape
            S0_var_h = ADVar(S0_val, requires_grad=True, name="S0")
            K_var_h = ADVar(self.K, requires_grad=True, name="K")
            r_var_h = ADVar(self.r, requires_grad=True, name="r")
            T_var_h = ADVar(self.T, requires_grad=True, name="T")

            # Build t_grid_ad for Hessian pass
            t_grid_ad_h = [ADVar(0.0, requires_grad=False)]
            for i in range(1, N + 1):
                t_grid_ad_h.append(ADVar(i, requires_grad=False) * (T_var_h / ADVar(N, requires_grad=False)))

            # Rebuild terminal condition on the fresh tape
            V_terminal_h = self._terminal_condition(K_var_h)
            V_h = V_terminal_h[1:-1]
            for n in range(N):
                t_current_ad_h = t_grid_ad_h[n + 1]
                sigma_local_h = self.sigma_loc(float(t_current_ad_h.val))
                a_L_h, b_L_h, c_L_h, a_R_h, b_R_h, c_R_h = self.build_tridiagonal_cn(
                    sigma_local_h, T_var_h / ADVar(N, requires_grad=False), r_var_h, K_var_h
                )
                V_h = self.cn_step(V_h, a_L_h, b_L_h, c_L_h, a_R_h, b_R_h, c_R_h, t_current_ad_h, r_var_h, K_var_h, T_var_h)

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

            # Edge-Pushing for full Hessian across S0, K, r, T, and all w_slices
            inputs = [S0_var_h, K_var_h, r_var_h, T_var_h] + self.w_slices
            hessian = algo4_adjlist(price_var_h, inputs)

            # Add readable variable labels
            var_names = ["S0", "K", "r", "T"] + [f"w{i}" for i in range(len(self.w_slices))]
            result["hessian_labels"] = var_names

            # Save full Hessian matrix and extract gamma if available
            result['hessian'] = hessian
            if hessian.shape[0] >= 1:
                result['gamma'] = hessian[0, 0]

        # === Runtime measurement ===
        t_end = time.perf_counter()
        time_sec = t_end - t_start
        result["runtime_sec"] = time_sec
        print(f"Runtime: {time_sec:10.2f} s")

        return result
