# pde_taylor_svi_dupire.py
"""
PDE + Taylor Expansion under SVI-Implied Vol Surface with Dupire Local Volatility

What this file does
-------------------
- Build a 1D Black–Scholes PDE in S-space with Crank–Nicolson scheme.
- Model the implied total variance surface w(k, T) using *raw SVI*:
      w(k,T) = a(T) + b(T) [ rho(T)(k - m(T)) + sqrt((k - m(T))^2 + eta(T)^2 ) ]
- Use (approximate) Dupire formula to obtain local volatility:
      sigma_loc(S, t) = sigma_loc(K=S, T_eff=t)
  where T_eff is used as time-to-maturity proxy inside the Dupire step.
- Run a single Taylor backpropagation on all seeds:
      seeds = [S0, K, r, T](+ [q] optional) + all SVI parameters (a_i,b_i,rho_i,m_i,eta_i)
  to obtain:
      * price
      * full gradient and Hessian in seed order
      * convenient Δ, Γ (w.r.t. S0)

Notes
-----
- This file is independent of your previous Dupire+flat-SVI implementation.
- We only focus on AD flow w.r.t. global variables and SVI parameters.
- Dupire derivatives (∂_T w, ∂_k w, ∂_{kk} w) are done via numeric finite
  differences in k,T, but still return ADVar so sensitivities flow through
  SVI parameters.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Callable
import time

# make sure we can import aad.core.*
sys.path.insert(0, str(Path(__file__).parent))

from aad.core.var import ADVar
from aad.core.tape import global_tape
from aad.core.taylor_backprop import taylor_backpropagate


# ============================== cubic spline helper ==============================

def _compute_spline_second_derivatives_AD(
    V: List[ADVar],
    S_grid: np.ndarray
) -> List[ADVar]:
    """
    Natural cubic spline second derivatives M_i for ADVar values V[i] on grid S_grid[i].

    Parameters
    ----------
    V : list[ADVar]
        Function values at S_grid.
    S_grid : np.ndarray
        1D grid (float) of same length as V.

    Returns
    -------
    M_vals : list[ADVar]
        Second derivatives at each node (natural spline, M_0 = M_n-1 = 0).
    """
    n = len(V)
    if n < 3:
        return [ADVar(0.0, requires_grad=False) for _ in range(n)]

    h = np.diff(S_grid)  # h[i] = S[i+1] - S[i]

    lambda_vals = []
    mu_vals = []
    d_vals = []

    for i in range(1, n - 1):
        h_im1 = h[i - 1]
        h_i = h[i]

        lam_i = h_im1 / (h_im1 + h_i)
        mu_i = h_i / (h_im1 + h_i)

        d_i = (ADVar(6.0, requires_grad=False) /
               ADVar(h_im1 + h_i, requires_grad=False)) * (
            (V[i + 1] - V[i]) / ADVar(h_i, requires_grad=False)
            - (V[i] - V[i - 1]) / ADVar(h_im1, requires_grad=False)
        )
        lambda_vals.append(lam_i)
        mu_vals.append(mu_i)
        d_vals.append(d_i)

    n_int = n - 2
    if n_int == 0:
        return [ADVar(0.0, requires_grad=False) for _ in range(n)]

    # a,b,c for tridiagonal system on interior points
    a = [ADVar(0.0, requires_grad=False)] + [
        ADVar(lam, requires_grad=False) for lam in lambda_vals
    ]
    b = [ADVar(2.0, requires_grad=False) for _ in range(n_int)]
    c = [ADVar(mu, requires_grad=False) for mu in mu_vals] + [
        ADVar(0.0, requires_grad=False)
    ]
    d_rhs = d_vals

    c_prime = [None] * n_int
    d_prime = [None] * n_int

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d_rhs[0] / b[0]

    for i in range(1, n_int):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n_int - 1 else ADVar(0.0, requires_grad=False)
        d_prime[i] = (d_rhs[i] - a[i] * d_prime[i - 1]) / denom

    M_int = [None] * n_int
    M_int[-1] = d_prime[-1]
    for i in range(n_int - 2, -1, -1):
        M_int[i] = d_prime[i] - c_prime[i] * M_int[i + 1]

    M_vals = [ADVar(0.0, requires_grad=False)] + M_int + [ADVar(0.0, requires_grad=False)]
    return M_vals


# ========================= SVI + Dupire PDE + Taylor =========================

class BS_PDE_Taylor_SVI:
    """
    Black–Scholes PDE in S-space with SVI implied vol surface + Dupire local volatility,
    and Taylor backpropagation over all seeds (global vars + SVI parameters).

    PDE:
        dV/dt + 0.5 sigma_loc^2(S,t) S^2 d2V/dS2 + (r - q) S dV/dS - r V = 0

    Scheme:
        - Crank–Nicolson with weight phi (default 0.5).
        - Uniform S-grid in [0, S_max_mult * K].
        - AD-aware coefficients via sigma_loc(S_i, t_k) from SVI + Dupire.
    """

    def __init__(self,
                 S0: float, K: float, T: float,
                 r,               # float or callable(t)->float
                 q=0.0,           # float or callable(t)->float
                 M: int = 101, N_base: int = 100,
                 S_max_mult: float = 3.0,
                 phi: float = 0.5):
        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.M = int(M)
        self.N_base = int(N_base)
        self.phi = float(phi)

        # unify r, q as callables
        if callable(r):
            self.r = r
        else:
            r_const = float(r)
            self.r = lambda t, _r=r_const: _r

        if callable(q):
            self.q = q
        else:
            q_const = float(q)
            self.q = lambda t, _q=q_const: _q

        # S-grid (uniform)
        S_min, S_max = 0.0, S_max_mult * self.K
        self.S_grid = np.linspace(S_min, S_max, self.M)
        self.dS = self.S_grid[1] - self.S_grid[0]

    # ---------------- terminal & boundary conditions (numeric) ----------------

    def _terminal_condition(self) -> np.ndarray:
        """European call payoff on the S-grid at t = T (numeric array)."""
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_left_numeric(self, t: float) -> float:
        """As S -> 0, call ≈ 0."""
        return 0.0

    def _boundary_right_numeric(self, t: float) -> float:
        """
        As S -> S_max, call ≈ S_max * e^{-q (T - t)} - K e^{-r (T - t)} (numeric).
        """
        tau = self.T - t
        r_now = self.r(t)
        q_now = self.q(t)
        S_max = self.S_grid[-1]
        return S_max * np.exp(-q_now * tau) - self.K * np.exp(-r_now * tau)

    # ---------------- SVI surface helpers (AD-aware parameters) ----------------

    @staticmethod
    def _svi_raw(k: ADVar, a: ADVar, b: ADVar, rho: ADVar,
                 m: ADVar, eta: ADVar) -> ADVar:
        """
        Raw SVI total variance for a fixed maturity:
            w(k) = a + b * [ rho*(k - m) + sqrt( (k - m)^2 + eta^2 ) ]
        """
        x = k - m
        sqrt_term = (x * x + eta * eta) ** ADVar(0.5, requires_grad=False)
        return a + b * (rho * x + sqrt_term)

    def _build_svi_param_AD(
        self,
        T_knots: np.ndarray,
        svi_param_vals: np.ndarray
    ):
        """
        Convert numeric SVI params into ADVars and prepare T-interpolation.

        svi_param_vals: shape (n_T, 5) with columns [a, b, rho, m, eta].

        Returns
        -------
        svi_param_lists: tuple of lists (a_vars, b_vars, rho_vars, m_vars, eta_vars)
        params_at_T(t): callable(float) -> (a_t, b_t, rho_t, m_t, eta_t) [ADVar]
        """
        T = np.asarray(T_knots, dtype=float)
        svi_param_vals = np.asarray(svi_param_vals, dtype=float)
        assert svi_param_vals.shape[0] == len(T) and svi_param_vals.shape[1] == 5

        a_vars: List[ADVar] = []
        b_vars: List[ADVar] = []
        rho_vars: List[ADVar] = []
        m_vars: List[ADVar] = []
        eta_vars: List[ADVar] = []

        for i, (a, b, rho, m, eta) in enumerate(svi_param_vals):
            a_vars.append(ADVar(float(a),   requires_grad=True, name=f"a_{i}"))
            b_vars.append(ADVar(float(b),   requires_grad=True, name=f"b_{i}"))
            rho_vars.append(ADVar(float(rho), requires_grad=True, name=f"rho_{i}"))
            m_vars.append(ADVar(float(m),   requires_grad=True, name=f"m_{i}"))
            eta_vars.append(ADVar(float(eta), requires_grad=True, name=f"eta_{i}"))

        def params_at_T(t: float):
            """
            Linearly interpolate (a,b,rho,m,eta) in T.

            t : float, used as time-to-maturity argument for SVI surface.
            """
            if t <= T[0]:
                return (a_vars[0], b_vars[0], rho_vars[0], m_vars[0], eta_vars[0])
            if t >= T[-1]:
                return (a_vars[-1], b_vars[-1], rho_vars[-1], m_vars[-1], eta_vars[-1])

            j = np.searchsorted(T, t)
            i = j - 1
            w = (t - T[i]) / (T[j] - T[i])
            w0 = 1.0 - w
            w1 = w

            w0_AD = ADVar(w0, requires_grad=False)
            w1_AD = ADVar(w1, requires_grad=False)

            a_t   = w0_AD * a_vars[i]   + w1_AD * a_vars[j]
            b_t   = w0_AD * b_vars[i]   + w1_AD * b_vars[j]
            rho_t = w0_AD * rho_vars[i] + w1_AD * rho_vars[j]
            m_t   = w0_AD * m_vars[i]   + w1_AD * m_vars[j]
            eta_t = w0_AD * eta_vars[i] + w1_AD * eta_vars[j]

            return a_t, b_t, rho_t, m_t, eta_t

        return (a_vars, b_vars, rho_vars, m_vars, eta_vars), params_at_T

    def _build_w_svi(
        self,
        T_knots: np.ndarray,
        svi_param_vals: np.ndarray
    ):
        """
        Build SVI total variance surface.

        Returns
        -------
        svi_param_lists : tuple of lists of ADVar
        params_at_T     : callable
        w_svi           : callable(k: ADVar, t: float) -> ADVar total variance
        """
        (a_vars, b_vars, rho_vars, m_vars, eta_vars), params_at_T = \
            self._build_svi_param_AD(T_knots, svi_param_vals)

        def w_svi(k: ADVar, t: float) -> ADVar:
            """SVI total variance w(k, t) with parameters interpolated in T."""
            a_t, b_t, rho_t, m_t, eta_t = params_at_T(t)
            return self._svi_raw(k, a_t, b_t, rho_t, m_t, eta_t)

        svi_param_lists = (a_vars, b_vars, rho_vars, m_vars, eta_vars)
        return svi_param_lists, params_at_T, w_svi

    # ---------------- Dupire local vol from SVI (numeric dk, dT) ----------------

    def _dupire_local_vol(
        self,
        S_val: float,
        t_current: float,
        w_svi: Callable[[ADVar, float], ADVar],
        dT: float = 1e-3,
        dk: float = 1e-3,
        eps: float = 1e-8
    ) -> ADVar:
        """
        Approximate Dupire local vol sigma_loc(S, t) from SVI total variance surface.

        Steps:
          - Build forward F(T) ≈ S0 * exp((r-q)*T_eff)
          - k = log(K/F(T_eff)) with K := S_val
          - Compute w, w_T, w_k, w_kk numerically in (k, T_eff), using ADVars
            so sensitivities w.r.t. SVI parameters propagate.

        Parameters
        ----------
        S_val : float
            Spot level on grid (used as strike K in Dupire formula).
        t_current : float
            PDE time variable in [0, T]. Here we interpret it as "effective maturity".
        """
        # Effective maturity for SVI surface
        T_eff = max(t_current, 1e-6)

        # Forward F(T_eff) ~ S0 * exp((r - q) * T_eff)
        r_T = self.r(0.0)
        q_T = self.q(0.0)
        F_T = self.S0 * np.exp((r_T - q_T) * T_eff)

        K_val = max(S_val, 1e-8)
        F_T = max(F_T, 1e-8)
        k0 = np.log(K_val / F_T)

        # Helper: evaluate w_svi at numeric (k,T) but ADVar in SVI params
        def w_at(k_float: float, T_float: float) -> ADVar:
            k_AD = ADVar(k_float, requires_grad=False)
            return w_svi(k_AD, T_float)

        # Total variance and derivatives
        w0 = w_at(k0, T_eff)
        w_T_plus = w_at(k0, T_eff + dT)
        w_T_minus = w_at(k0, T_eff - dT)
        dw_dT = (w_T_plus - w_T_minus) / ADVar(2.0 * dT, requires_grad=False)

        w_k_plus = w_at(k0 + dk, T_eff)
        w_k_minus = w_at(k0 - dk, T_eff)
        dw_dk = (w_k_plus - w_k_minus) / ADVar(2.0 * dk, requires_grad=False)
        d2w_dk2 = (w_k_plus - ADVar(2.0, requires_grad=False) * w0 + w_k_minus) / \
                  ADVar(dk * dk, requires_grad=False)

        # Dupire denominator (simplified standard variant)
        k_AD = ADVar(k0, requires_grad=False)
        one = ADVar(1.0, requires_grad=False)
        quarter = ADVar(0.25, requires_grad=False)
        half = ADVar(0.5, requires_grad=False)
        eps_AD = ADVar(eps, requires_grad=False)

        w_safe = w0 + eps_AD
        term1 = (one - k_AD * dw_dk / w_safe)
        denom = term1 * term1 + quarter * dw_dk * dw_dk - half * d2w_dk2
        denom_safe = denom + eps_AD

        sigma_sq = dw_dT / denom_safe
        sigma_sq_safe = sigma_sq + eps_AD
    
        # ---- 安全 clip，避免出现负的 sigma^2 导致 complex ----
        sigma_sq_val = float(sigma_sq_safe.val)
        if sigma_sq_val <= eps:
            # 如果数值上是负的或太接近 0，就直接 clip 到一个小正数
            sigma_sq_safe = ADVar(eps, requires_grad=False)
        # ----------------------------------------------

        sigma_loc = sigma_sq_safe ** ADVar(0.5, requires_grad=False)
        return sigma_loc
    
        # ---------------- tridiagonal solver & CN step ----------------

    def tridiag_solve(self, a, b, c, d):
        """Thomas algorithm with ADVar entries."""
        n = len(d)
        c_prime = [None] * n
        d_prime = [None] * n

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i - 1]
            c_prime[i] = c[i] / denom if i < n - 1 else ADVar(0.0)
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

        x = [None] * n
        x[-1] = d_prime[-1]
        for i in range(n - 2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i + 1]
        return x

    def _boundary_right_AD(self,
                           t_ad: ADVar,
                           r_var: ADVar,
                           q_var: ADVar,
                           K_var: ADVar,
                           T_var: ADVar) -> ADVar:
        """
        Upper boundary for a European call in AD form:
          V(S_max, t) ≈ S_max * exp(-q * (T - t)) - K * exp(-r * (T - t))
        """
        S_max = self.S_grid[-1]
        tau = T_var - t_ad
        return ADVar(S_max, requires_grad=False) * (-q_var * tau).exp() \
             - K_var * (-r_var * tau).exp()

    def build_cn_with_svi(self,
                          dt: ADVar,
                          t_current: ADVar,
                          r_var: ADVar,
                          q_var: ADVar,
                          w_svi: Callable[[ADVar, float], ADVar]):
        """
        Build CN tri-diagonal coefficients at time t_current using
        sigma_loc(S_i, t_current) from SVI + Dupire.
        """
        n = self.M - 2
        dS = self.dS

        c0 = ADVar(0.0, requires_grad=False)
        c1 = ADVar(1.0, requires_grad=False)
        c2 = ADVar(2.0, requires_grad=False)
        cneg = ADVar(-1.0, requires_grad=False)
        phi = ADVar(self.phi, requires_grad=False)
        one_minus_phi = c1 + (cneg * phi)
        dS_sq = ADVar(dS * dS, requires_grad=False)
        dS_2 = ADVar(2.0 * dS, requires_grad=False)

        a_L: List[ADVar] = []
        b_L: List[ADVar] = []
        c_L: List[ADVar] = []
        a_R: List[ADVar] = []
        b_R: List[ADVar] = []
        c_R: List[ADVar] = []

        t_float = float(t_current.val)

        for i in range(n):
            S_i_val = self.S_grid[i + 1]
            S_i_var = ADVar(S_i_val, requires_grad=False)

            sigma_i = self._dupire_local_vol(S_i_val, t_float, w_svi)

            alpha = (sigma_i * sigma_i * S_i_var * S_i_var / c2) / dS_sq
            beta = ((r_var - q_var) * S_i_var) / dS_2
            gamma = cneg * r_var

            l_i = alpha - beta
            c_i = (cneg * c2) * alpha + gamma
            u_i = alpha + beta

            a_L.append(c0 if i == 0     else (cneg * phi * dt * l_i))
            b_L.append(      c1 - phi * dt * c_i)
            c_L.append(c0 if i == n - 1 else (cneg * phi * dt * u_i))

            a_R.append(c0 if i == 0     else (one_minus_phi * dt * l_i))
            b_R.append(      c1 + one_minus_phi * dt * c_i)
            c_R.append(c0 if i == n - 1 else (one_minus_phi * dt * u_i))

        return a_L, b_L, c_L, a_R, b_R, c_R

    def cn_step_AD(self,
                   V: List[ADVar],
                   a_L, b_L, c_L, a_R, b_R, c_R,
                   t_current: ADVar,
                   r_var: ADVar, q_var: ADVar,
                   K_var: ADVar, T_var: ADVar) -> List[ADVar]:
        """One CN step with AD right boundary."""
        n = self.M - 2
        rhs: List[ADVar] = [None] * n
        for i in range(n):
            if i == 0:
                V_left = ADVar(self._boundary_left_numeric(float(t_current.val)),
                               requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i + 1] - a_R[i] * V_left
            elif i == n - 1:
                V_right = self._boundary_right_AD(t_current, r_var, q_var, K_var, T_var)
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] + c_R[i] * V[i + 1]
        return self.tridiag_solve(a_L, b_L, c_L, rhs)

    # ---------------- main API: SVI all-seeds solver ----------------

    def solve_pde_taylor_svi_allseeds(
            self,
            S0_val: float,
            T_knots: np.ndarray,
            svi_param_vals: np.ndarray,
            verbose: bool = False,
            make_q_seed: bool = False
        ) -> Dict:
        """
        Main entry:
    
          seeds = [S0, K, r, T] (+ [q] optional) + all SVI params (a_i,b_i,rho_i,m_i,eta_i)
    
        Output dict:
          - price
          - gradient, hessian  (numpy arrays) in seed order
          - seed_names
          - delta, gamma (w.r.t. S0)
          - tape_nodes
          - time_ms: total time
          - build_ms: seeds + SVI + dt 构建时间
          - pde_ms: PDE stepping + spline readout 时间
          - backprop_ms: Taylor 反传时间
        """
        t0_all = time.perf_counter()
        global_tape.reset()
    
        # ================== 1) 构建 seeds + SVI surface + dt ==================
        t_build0 = time.perf_counter()
    
        # ---- global seeds ----
        S0_var = ADVar(float(S0_val), requires_grad=True,  name="S0")
        K_var  = ADVar(float(self.K),  requires_grad=True,  name="K")
        r_var  = ADVar(float(self.r(0.0) if callable(self.r) else self.r),
                       requires_grad=True, name="r")
        T_var  = ADVar(float(self.T),  requires_grad=True,  name="T")
        q_var  = ADVar(float(self.q(0.0) if callable(self.q) else self.q),
                       requires_grad=bool(make_q_seed), name="q")
    
        # SVI surface (Dupire total variance w_svi)
        svi_param_lists, params_at_T, w_svi = self._build_w_svi(T_knots, svi_param_vals)
        a_vars, b_vars, rho_vars, m_vars, eta_vars = svi_param_lists
    
        # time grid via AD T_var (same style as Dupire-flat 实现)
        N = int(self.N_base)
        dt = T_var / ADVar(float(N), requires_grad=False)
    
        build_ms = (time.perf_counter() - t_build0) * 1000.0
    
        # ================== 2) PDE stepping + spline readout ==================
        t_pde0 = time.perf_counter()
    
        # terminal condition at t = T (numeric → AD constants)
        V_terminal = self._terminal_condition()
        V: List[ADVar] = [ADVar(float(v), requires_grad=False) for v in V_terminal[1:-1]]
    
        # backward stepping: t_k = k*dt, k = N-1..0
        for k in range(N - 1, -1, -1):
            t_k = ADVar(float(k), requires_grad=False) * dt
            a_L, b_L, c_L, a_R, b_R, c_R = self.build_cn_with_svi(
                dt, t_k, r_var, q_var, w_svi
            )
            V = self.cn_step_AD(
                V, a_L, b_L, c_L, a_R, b_R, c_R,
                t_k, r_var, q_var, K_var, T_var
            )
    
        # ---------- cubic spline readout at S0 (AD) ----------
        _const = lambda v: ADVar(v, requires_grad=False)
        S_interior = self.S_grid[1:-1]
        n_int = len(V)
    
        M_vals = _compute_spline_second_derivatives_AD(V, S_interior)
    
        idx = np.searchsorted(S_interior, S0_val)
        if idx == 0:
            idx = 1
        elif idx >= n_int:
            idx = n_int - 1
        i = idx - 1
        S_i  = S_interior[i]
        S_i1 = S_interior[i + 1]
        V_i  = V[i]
        V_i1 = V[i + 1]
        M_i  = M_vals[i]
        M_i1 = M_vals[i + 1]
        h    = S_i1 - S_i
    
        S_i_var  = _const(S_i)
        S_i1_var = _const(S_i1)
        h_var    = _const(h)
    
        A  = (S_i1_var - S0_var) / h_var
        B  = (S0_var - S_i_var) / h_var
        A3 = A * A * A
        B3 = B * B * B
        h2_over_6 = h_var * h_var / ADVar(6.0, requires_grad=False)
    
        price_var = (
            A * V_i + B * V_i1 +
            (A3 - A) * h2_over_6 * M_i +
            (B3 - B) * h2_over_6 * M_i1
        )
        # ---------- spline readout end ----------
    
        pde_ms = (time.perf_counter() - t_pde0) * 1000.0
    
        # ================== 3) Taylor backprop ==================
        t_back0 = time.perf_counter()
    
        svi_flat = a_vars + b_vars + rho_vars + m_vars + eta_vars
        seeds = [S0_var, K_var, r_var, T_var] + ([q_var] if make_q_seed else []) + svi_flat
        grad, H = taylor_backpropagate(seeds, target=price_var)
    
        backprop_ms = (time.perf_counter() - t_back0) * 1000.0
    
        # ================== 收尾 & 输出 ==================
        price = float(price_var.val)
        delta = float(grad[0])
        gamma = float(H[0, 0])
    
        seed_names = (
            ["S0", "K", "r", "T"] +
            (["q"] if make_q_seed else []) +
            [v.name for v in svi_flat]
        )
    
        total_ms = (time.perf_counter() - t0_all) * 1000.0
    
        out = {
            "price": price,
            "gradient": grad,
            "hessian": H,
            "seed_names": seed_names,
            "delta": delta,
            "gamma": gamma,
            "tape_nodes": len(global_tape.nodes),
            "time_ms": total_ms,
            "build_ms": build_ms,
            "pde_ms": pde_ms,
            "backprop_ms": backprop_ms,
            "note": "SVI+Dupire local vol; seeds=[S0,K,r,T](+q)+SVI params; cubic spline in S",
        }
    
        if verbose:
            print(f"[SVI Dupire all-seeds] Price={price:.8f}  Δ={delta:.6f}  Γ={gamma:.6e}")
            print("  seed order:", seed_names)
            print(f"  Build seeds/SVI+dt : {build_ms:.1f} ms")
            print(f"  PDE stepping+read  : {pde_ms:.1f} ms")
            print(f"  Taylor backprop    : {backprop_ms:.1f} ms")
            print(f"  Tape nodes         : {out['tape_nodes']:,}")
            print(f"  Total              : {total_ms:.1f} ms")
    
        return out


# ================================== quick self-check ==================================

if __name__ == "__main__":
    # Simple sanity run with made-up SVI parameters (NOT calibrated to market).
    S0, K, T = 100.0, 100.0, 1.0
    r, q = 0.05, 0.0

    # Example maturities and SVI params: [a, b, rho, m, eta]
    T_knots = np.array([0.25, 0.5, 1.0], dtype=float)
    svi_params = np.array([
        [0.01,  0.20, -0.30, 0.00, 0.20],
        [0.015, 0.22, -0.25, 0.00, 0.25],
        [0.020, 0.25, -0.20, 0.00, 0.30],
    ])

    solver = BS_PDE_Taylor_SVI(S0, K, T, r=r, q=q, M=101, N_base=80, S_max_mult=3.0)
    out = solver.solve_pde_taylor_svi_allseeds(
        S0_val=S0,
        T_knots=T_knots,
        svi_param_vals=svi_params,
        verbose=True,
        make_q_seed=False
    )

    print("\n=== Summary ===")
    print(f"Price: {out['price']:.8f}")
    print(f"Delta: {out['delta']:.6f}")
    print(f"Gamma: {out['gamma']:.6e}")
    print(f"Tape nodes: {out['tape_nodes']:,}")
    print(f"Time (ms): {out['time_ms']:.2f}")
