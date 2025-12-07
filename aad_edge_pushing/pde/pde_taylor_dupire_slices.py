# pde_taylor_dupire_slices.py
"""
Dupire-style PDE + Taylor Expansion with flat time slices of total variance w(T).

- Inputs:   T_knots = [T0, T1, ..., Tm] (increasing)
            w_slice_vals = [w(T0), ..., w(Tm)]  where w(T)=sigma^2 * T (total variance)

- Model:    Build w(t) by linear interpolation on [T_i, T_{i+1}],
            sigma_loc(t) = sqrt(max(dw/dt, eps)).

- Scheme:   Black–Scholes PDE in S-space, Crank–Nicolson with weight phi (default 0.5),
            uniform S-grid in [0, S_max_mult * K].

- AD seeds: ONE-SHOT Taylor backprop with seeds:
            [S0, K, r, T] (+ optional q) + [w_0, ..., w_{m-1}]
            Returns full gradient/Hessian including all cross terms.
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Callable
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad.core.var import ADVar
from aad.core.tape import global_tape
from aad.core.taylor_backprop import taylor_backpropagate


# ============================== helpers ==============================

def build_w_t_linear_AD(T_knots: np.ndarray, w_vars: List[ADVar]) -> Tuple[Callable[[float], ADVar],
                                                                           Callable[[float], ADVar]]:
    """
    Linear interpolation of total variance w(t) with ADVar knots.
    Returns:
      w_t(t):   ADVar
      dw_dt(t): ADVar (piecewise-constant slope per sub-interval)
    """
    T = np.asarray(T_knots, dtype=float)
    wv = list(w_vars)
    assert len(T) == len(wv) and len(T) >= 2

    def w_t(t: float) -> ADVar:
        if t <= T[0]:
            return wv[0]
        if t >= T[-1]:
            return wv[-1]
        j = np.searchsorted(T, t)
        i = j - 1
        w = (t - T[i]) / (T[j] - T[i])
        return ADVar(1.0 - w, requires_grad=False) * wv[i] + ADVar(w, requires_grad=False) * wv[j]

    def dw_dt(t: float) -> ADVar:
        if t <= T[0]:
            denom = T[1] - T[0]
            return (wv[1] - wv[0]) / ADVar(denom, requires_grad=False)
        if t >= T[-1]:
            denom = T[-1] - T[-2]
            return (wv[-1] - wv[-2]) / ADVar(denom, requires_grad=False)
        j = np.searchsorted(T, t)
        i = j - 1
        denom = T[j] - T[i]
        return (wv[j] - wv[i]) / ADVar(denom, requires_grad=False)

    return w_t, dw_dt


def sigma_from_w_slope(dw_dt_val: ADVar, eps: float = 1e-12) -> ADVar:
    """
    Dupire local volatility proxy:
      sigma_loc(t) = sqrt(max(dw/dt, eps))
    Note: We keep it as ADVar so sensitivities w.r.t. w_i flow through.
    """
    return (dw_dt_val + ADVar(eps, requires_grad=False)) ** ADVar(0.5, requires_grad=False)


# ======================= main Taylor PDE solver =======================

class BS_PDE_Taylor_DupireSlices:
    """
    Black–Scholes PDE (S-space) + Taylor backprop under Dupire-style total-variance slices.

    Grid & scheme:
      - Uniform S-grid in [0, S_max_mult * K]
      - Crank–Nicolson with weight phi (default 0.5)

    Readout:
      - Linear interpolation on S-grid using ADVar(S0)
      - One Taylor backprop over seeds [S0, K, r, T](+q) + [w_i] gives full gradient/Hessian
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

    # ---------------- tridiagonal & CN step (numeric core) ----------------

    def tridiag_solve(self, a: List[ADVar], b: List[ADVar], c: List[ADVar],
                      d: List[ADVar]) -> List[ADVar]:
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

    # ---------------- AD-aware boundaries & assembly ----------------

    def _boundary_right_AD(self, t_ad: ADVar, r_var: ADVar, q_var: ADVar,
                           K_var: ADVar, T_var: ADVar) -> ADVar:
        """
        Upper boundary for a European call:
          V(S_max, t) ≈ S_max * exp(-q * (T - t)) - K * exp(-r * (T - t))
        Implemented in AD so derivatives w.r.t. r, q, K, T propagate.
        """
        S_max = self.S_grid[-1]
        tau = T_var - t_ad
        # If your ADVar does not implement ".exp()", replace by an AD-friendly exp wrapper.
        return ADVar(S_max, requires_grad=False) * (-q_var * tau).exp() \
             - K_var * (-r_var * tau).exp()

    def build_cn_with_params(self,
                             sigma_t: Callable[[float], ADVar],
                             dt: ADVar,               # ADVar Δt = T_var / N
                             t_current: ADVar,        # ADVar time = k*dt
                             r_var: ADVar, q_var: ADVar):
        """
        Build CN tri-diagonal coefficients at time t_current using sigma(t_current),
        with r, q as ADVars (constant across the step).
        Returns lists of ADVar: a_L, b_L, c_L, a_R, b_R, c_R (length n=M-2)
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

        # sigma_t takes a float to decide which slice to use; sensitivity flows via w_i.
        sigma_var = sigma_t(float(t_current.val))

        a_L, b_L, c_L = [], [], []
        a_R, b_R, c_R = [], [], []

        for i in range(n):
            S_i_var = ADVar(self.S_grid[i + 1], requires_grad=False)
            alpha = (sigma_var * sigma_var * S_i_var * S_i_var / c2) / dS_sq
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

    def cn_step_AD(self, V: List[ADVar],
                   a_L, b_L, c_L, a_R, b_R, c_R,
                   t_current: ADVar, r_var: ADVar, q_var: ADVar,
                   K_var: ADVar, T_var: ADVar) -> List[ADVar]:
        """One CN step with AD right boundary."""
        n = self.M - 2
        rhs = [None] * n
        for i in range(n):
            if i == 0:
                V_left = ADVar(0.0, requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i + 1] - a_R[i] * V_left
            elif i == n - 1:
                V_right = self._boundary_right_AD(t_current, r_var, q_var, K_var, T_var)
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] + c_R[i] * V[i + 1]
        return self.tridiag_solve(a_L, b_L, c_L, rhs)

    # ---------------- all-seeds main API ----------------

    def solve_pde_taylor_w_slices_allseeds(
        self,
        S0_val: float,
        T_knots: np.ndarray,
        w_slice_vals: np.ndarray,
        verbose: bool = False,
        make_q_seed: bool = False
    ) -> Dict:
        """
        Dupire + ALL seeds:
          seeds = [S0, K, r, T] (+ [q] optional) + [w_i]
        One Taylor backprop returns price, full gradient/Hessian and convenient Greek views.
        """
        t0_all = time.perf_counter()
        global_tape.reset()

        # ---- AD seeds ----
        S0_var = ADVar(float(S0_val), requires_grad=True,  name="S0")
        K_var  = ADVar(float(self.K),  requires_grad=True,  name="K")
        r_var  = ADVar(float(self.r(0.0) if callable(self.r) else self.r),
                       requires_grad=True, name="r")
        T_var  = ADVar(float(self.T),  requires_grad=True,  name="T")
        q_var  = ADVar(float(self.q(0.0) if callable(self.q) else self.q),
                       requires_grad=bool(make_q_seed), name="q")

        w_slice_vals = np.asarray(w_slice_vals, dtype=float)
        w_vars = [ADVar(float(w), requires_grad=True, name=f"w_{i}") for i, w in enumerate(w_slice_vals)]

        # ---- w(t) & sigma(t) ----
        T_knots = np.asarray(T_knots, dtype=float)
        assert len(T_knots) == len(w_vars) and len(T_knots) >= 2
        _, dwdt_t = build_w_t_linear_AD(T_knots, w_vars)
        def sigma_t(t: float) -> ADVar:
            return sigma_from_w_slope(dwdt_t(t))

        # ---- time grid via AD T_var ----
        N = int(self.N_base)
        dt = T_var / ADVar(float(N), requires_grad=False)  # ADVar Δt so price depends on T

        # terminal condition (numeric payoff grid; K sensitivity enters via boundary and time stepping)
        V_terminal = np.maximum(self.S_grid - float(K_var.val), 0.0)
        V = [ADVar(float(v), requires_grad=False) for v in V_terminal[1:-1]]

        # backward stepping: t_k = k * dt, k = N-1..0
        for k in range(N - 1, -1, -1):
            t_k = ADVar(float(k), requires_grad=False) * dt
            a_L, b_L, c_L, a_R, b_R, c_R = self.build_cn_with_params(
                sigma_t, dt, t_k, r_var, q_var
            )
            V = self.cn_step_AD(V, a_L, b_L, c_L, a_R, b_R, c_R, t_k, r_var, q_var, K_var, T_var)
        # --- safe AD constants helper ---
        _const = lambda v: ADVar(v, requires_grad=False)

        # ---- AD readout at S0: linear interpolation using S0_var ----
        S_interior = self.S_grid[1:-1]
        idx = np.searchsorted(S_interior, S0_val)
        idx = max(1, min(idx, len(V) - 1))
        S1, S2 = S_interior[idx - 1], S_interior[idx]
        S1c = _const(S1)
        S2c = _const(S2)
        den = _const(S2 - S1)
        wL = (S2c - S0_var) / den
        wR = (S0_var - S1c) / den
        price_var = wL * V[idx - 1] + wR * V[idx]

        # ---- one-shot Taylor backprop ----
        seeds = [S0_var, K_var, r_var, T_var] + ([q_var] if make_q_seed else []) + w_vars
        grad, H = taylor_backpropagate(seeds, target=price_var)

        # ---- unpack convenient Greeks ----
        price = float(price_var.val)

        # index helpers
        off = 4 + (1 if make_q_seed else 0)
        vega_w = np.asarray(grad[off:], dtype=float)                  # dV/dw_i
        volga_w = np.asarray(H[off:, off:], dtype=float)             # d2V/(dw_i dw_j)
        vanna_w = np.asarray(H[0, off:], dtype=float)                # d2V/(dS0 dw_i)
        delta = float(H[0, 0] * 0 + grad[0])                         # dV/dS0
        gamma = float(H[0, 0])                                       # d2V/dS0^2

        out = {
            "price": price,
            "gradient": grad,          # full gradient in seed order
            "hessian": H,              # full Hessian in seed order
            "seed_names":
                (["S0", "K", "r", "T"] + (["q"] if make_q_seed else []) +
                 [sv.name for sv in w_vars]),
            # convenient reads
            "delta": delta,
            "gamma": gamma,
            "vega_w": vega_w,
            "volga_w": volga_w,
            "vanna_w": vanna_w,
            "tape_nodes": len(global_tape.nodes),
            "time_ms": (time.perf_counter() - t0_all) * 1000.0,
            "note": "Dupire sigma(t)=sqrt(max(dw/dt,eps)); seeds=[S0,K,r,T](+q)+w_i"
        }

        if verbose:
            print(f"[Dupire all-seeds] Price={price:.8f}  Δ={delta:.6f}  Γ={gamma:.6e}")
            print("  seed order:", out["seed_names"])
            print("  vega_w  :", vega_w)
            print("  vanna_w :", vanna_w)
            print("  volga_w :\n", volga_w)
            print(f"  Tape nodes: {out['tape_nodes']:,}, Time: {out['time_ms']:.1f} ms")

        return out


# =================== Finite Difference baselines & compare ===================

def _build_numeric_sigma_t_from_w(T_knots: np.ndarray, w_vals: np.ndarray):
    """
    Numeric helpers from (T_knots, w(T)):
      w_num(t): linear interpolation
      dwdt_num(t): piecewise-constant slope
      sigma_num(t): sqrt(max(dwdt, eps))
    """
    T = np.asarray(T_knots, dtype=float)
    W = np.asarray(w_vals, dtype=float)
    assert len(T) == len(W) and len(T) >= 2
    eps = 1e-12

    def w_num(t: float) -> float:
        if t <= T[0]:   return float(W[0])
        if t >= T[-1]:  return float(W[-1])
        j = np.searchsorted(T, t); i = j - 1
        w = (t - T[i]) / (T[j] - T[i])
        return float((1.0 - w) * W[i] + w * W[j])

    def dwdt_num(t: float) -> float:
        if t <= T[0]:
            return float((W[1] - W[0]) / (T[1] - T[0]))
        if t >= T[-1]:
            return float((W[-1] - W[-2]) / (T[-1] - T[-2]))
        j = np.searchsorted(T, t); i = j - 1
        return float((W[j] - W[i]) / (T[j] - T[i]))

    def sigma_num(t: float) -> float:
        return float(np.sqrt(max(dwdt_num(t), eps)))

    return w_num, dwdt_num, sigma_num


def price_dupire_numeric(solver: "BS_PDE_Taylor_DupireSlices",
                         S0_val: float,
                         T_knots: np.ndarray,
                         w_slice_vals: np.ndarray) -> float:
    """
    Pure numeric pricing (no AD backprop), used for FD baselines.
    """
    _, _, sigma_num = _build_numeric_sigma_t_from_w(T_knots, w_slice_vals)

    def sigma_t_AD(t: float) -> ADVar:
        return ADVar(sigma_num(t), requires_grad=False)

    t_grid = np.linspace(0.0, solver.T, solver.N_base + 1)
    dt_val = t_grid[1] - t_grid[0]
    dt = ADVar(dt_val, requires_grad=False)

    V_terminal = np.maximum(solver.S_grid - solver.K, 0.0)
    V = [ADVar(float(v), requires_grad=False) for v in V_terminal[1:-1]]

    for n in range(solver.N_base - 1, -1, -1):
        t_current = t_grid[n]
        a_L, b_L, c_L, a_R, b_R, c_R = solver.build_cn_with_params(
            sigma_t=lambda _: sigma_t_AD(t_current),
            dt=dt,
            t_current=ADVar(0.0, requires_grad=False),  # dummy (not used by numeric boundary)
            r_var=ADVar(solver.r(t_current), requires_grad=False),
            q_var=ADVar(solver.q(t_current), requires_grad=False),
        )
        # numeric right boundary (float) for the baseline:
        nint = solver.M - 2
        rhs = [None] * nint
        for i in range(nint):
            if i == 0:
                V_left = ADVar(0.0, requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i + 1] - a_R[i] * V_left
            elif i == nint - 1:
                tau = solver.T - t_current
                V_right = ADVar(
                    solver.S_grid[-1] * np.exp(-solver.q(t_current) * tau)
                    - solver.K * np.exp(-solver.r(t_current) * tau),
                    requires_grad=False
                )
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] + c_R[i] * V[i + 1]
        V = solver.tridiag_solve(a_L, b_L, c_L, rhs)

    S = solver.S_grid[1:-1]
    idx = np.searchsorted(S, S0_val)
    idx = max(1, min(idx, len(V) - 1))
    S1, S2 = S[idx - 1], S[idx]
    w = (S0_val - S1) / (S2 - S1)
    price = (1.0 - w) * V[idx - 1].val + w * V[idx].val
    return float(price)


def fd_grad_hess_w_slices(solver: "BS_PDE_Taylor_DupireSlices",
                          S0_val: float,
                          T_knots: np.ndarray,
                          w_slice_vals: np.ndarray,
                          h: float = 1e-4):
    """
    Central-difference FD for grad/Hessian w.r.t. w_i (used in comparisons).
    """
    t0 = time.perf_counter()
    base = np.asarray(w_slice_vals, dtype=float)
    n = len(base)

    grad = np.zeros(n)
    for i in range(n):
        x = base.copy(); x[i] += h
        vp = price_dupire_numeric(solver, S0_val, T_knots, x)
        x = base.copy(); x[i] -= h
        vm = price_dupire_numeric(solver, S0_val, T_knots, x)
        grad[i] = (vp - vm) / (2.0 * h)

    H = np.zeros((n, n))
    f0 = price_dupire_numeric(solver, S0_val, T_knots, base)
    for i in range(n):
        x = base.copy(); x[i] += h
        fp = price_dupire_numeric(solver, S0_val, T_knots, x)
        x = base.copy(); x[i] -= h
        fm = price_dupire_numeric(solver, S0_val, T_knots, x)
        H[i, i] = (fp - 2.0 * f0 + fm) / (h * h)

    for i in range(n):
        for j in range(i + 1, n):
            x = base.copy(); x[i] += h; x[j] += h
            f_pp = price_dupire_numeric(solver, S0_val, T_knots, x)
            x = base.copy(); x[i] += h; x[j] -= h
            f_pm = price_dupire_numeric(solver, S0_val, T_knots, x)
            x = base.copy(); x[i] -= h; x[j] += h
            f_mp = price_dupire_numeric(solver, S0_val, T_knots, x)
            x = base.copy(); x[i] -= h; x[j] -= h
            f_mm = price_dupire_numeric(solver, S0_val, T_knots, x)
            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
            H[i, j] = H[j, i] = H_ij

    elapsed = (time.perf_counter() - t0) * 1000.0
    return grad, H, elapsed


def _delta_fd_price_numeric(solver, S0_val, T_knots, w_slice_vals, hS):
    """Delta via central difference on pure-numeric prices."""
    vp = price_dupire_numeric(solver, S0_val + hS, T_knots, w_slice_vals)
    vm = price_dupire_numeric(solver, S0_val - hS, T_knots, w_slice_vals)
    return (vp - vm) / (2.0 * hS)


def fd_gamma_and_vanna_numeric(solver: "BS_PDE_Taylor_DupireSlices",
                               S0_val: float,
                               T_knots: np.ndarray,
                               w_slice_vals: np.ndarray,
                               hS: float = None,
                               hW: float = 1e-4):
    """
    FD for:
      gamma     = d2V/dS0^2 (S-direction 2nd central diff)
      vanna[i]  = d/dw_i (dV/dS0) via central diff on Delta over w_i
    """
    base = np.asarray(w_slice_vals, dtype=float)
    if hS is None:
        hS = solver.dS if getattr(solver, "dS", None) else max(1e-2 * float(S0_val), 1e-4)

    Vp = price_dupire_numeric(solver, S0_val + hS, T_knots, base)
    V0 = price_dupire_numeric(solver, S0_val,         T_knots, base)
    Vm = price_dupire_numeric(solver, S0_val - hS, T_knots, base)
    gamma = (Vp - 2.0 * V0 + Vm) / (hS * hS)

    n = len(base)
    vanna = np.zeros(n)
    for i in range(n):
        x = base.copy(); x[i] += hW
        delta_p = _delta_fd_price_numeric(solver, S0_val, T_knots, x, hS)
        x = base.copy(); x[i] -= hW
        delta_m = _delta_fd_price_numeric(solver, S0_val, T_knots, x, hS)
        vanna[i] = (delta_p - delta_m) / (2.0 * hW)

    return float(gamma), vanna


def compare_dupire_taylor_vs_fd(solver_kwargs: dict,
                                S0_val: float,
                                T_knots: np.ndarray,
                                w_slice_vals: np.ndarray,
                                hW: float = 1e-4,
                                hS: float = None,
                                verbose: bool = True):
    """
    Unified comparison:
      - Taylor (Dupire, all-seeds): solve_pde_taylor_w_slices_allseeds
      - FD baselines: grad/H wrt w_i + Gamma/Vanna + Price
    """
    # Taylor
    solver_aad = BS_PDE_Taylor_DupireSlices(**solver_kwargs)
    t0 = time.perf_counter()
    aad = solver_aad.solve_pde_taylor_w_slices_allseeds(S0_val, T_knots, w_slice_vals, verbose=False)
    aad_ms = (time.perf_counter() - t0) * 1000.0

    g_aad   = np.asarray(aad["vega_w"], dtype=float)
    H_aad   = np.asarray(aad["volga_w"], dtype=float)
    vanna_a = np.asarray(aad["vanna_w"], dtype=float)
    gamma_a = float(aad["gamma"])
    price_a = float(aad["price"])

    # FD
    solver_fd = BS_PDE_Taylor_DupireSlices(**solver_kwargs)
    t0 = time.perf_counter()
    g_fd, H_fd, fd_ms = fd_grad_hess_w_slices(solver_fd, S0_val, T_knots, w_slice_vals, h=hW)
    gamma_fd, vanna_fd = fd_gamma_and_vanna_numeric(solver_fd, S0_val, T_knots, w_slice_vals,
                                                    hS=hS, hW=hW)
    price_fd = price_dupire_numeric(solver_fd, S0_val, T_knots, w_slice_vals)

    # metrics
    eps = 1e-12
    def _vec_metrics(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        abs_ = np.abs(a - b)
        rmse = float(np.sqrt(np.mean(abs_**2)))
        rel  = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + eps))
        mxa  = float(np.max(abs_))
        return rmse, rel, mxa

    grad_rmse, grad_rel, grad_max = _vec_metrics(g_aad, g_fd)
    H_rmse, H_rel, H_max          = _vec_metrics(H_aad.ravel(), H_fd.ravel())
    vanna_rmse, vanna_rel, vanna_max = _vec_metrics(vanna_a, vanna_fd)
    gamma_abs = abs(gamma_a - gamma_fd); gamma_rel = gamma_abs / (abs(gamma_fd) + eps)
    price_abs = abs(price_a - price_fd); price_rel = price_abs / (abs(price_fd) + eps)

    res = {
        "price": {"taylor": price_a, "fd": price_fd, "abserr": price_abs, "relerr": price_rel},
        "grad":  {"taylor": g_aad, "fd": g_fd, "rmse": grad_rmse, "relerr": grad_rel, "maxabs": grad_max},
        "hess":  {"taylor": H_aad, "fd": H_fd, "rmse": H_rmse, "relerr": H_rel, "maxabs": H_max},
        "vanna": {"taylor": vanna_a, "fd": vanna_fd, "rmse": vanna_rmse, "relerr": vanna_rel, "maxabs": vanna_max},
        "gamma": {"taylor": gamma_a, "fd": gamma_fd, "abserr": gamma_abs, "relerr": gamma_rel},
        "timing_ms": {"taylor": aad_ms, "fd": fd_ms},
        "speedup_fd_over_taylor": (fd_ms / aad_ms) if aad_ms > 0 else float("inf"),
    }

    if verbose:
        pretty_print_dupire_compare("Dupire: Taylor vs FD", res)

    return res


def pretty_print_dupire_compare(title: str, res: dict):
    t = res["timing_ms"]; spd = res["speedup_fd_over_taylor"]
    lines = [
        f"=== {title} ===",
        f" Price           : Taylor={res['price']['taylor']:.8f} | FD={res['price']['fd']:.8f} | Abs={res['price']['abserr']:.3e} | Rel={res['price']['relerr']:.3e}",
        f" Grad (dV/dw_i)  : RMSE={res['grad']['rmse']:.3e} | Rel={res['grad']['relerr']:.3e} | MaxAbs={res['grad']['maxabs']:.3e}",
        f" Hess (d2/dw^2)  : RMSE={res['hess']['rmse']:.3e} | Rel={res['hess']['relerr']:.3e} | MaxAbs={res['hess']['maxabs']:.3e}",
        f" Vanna (S0×w)    : RMSE={res['vanna']['rmse']:.3e} | Rel={res['vanna']['relerr']:.3e} | MaxAbs={res['vanna']['maxabs']:.3e}",
        f" Gamma (S0)      : Abs={res['gamma']['abserr']:.3e} | Rel={res['gamma']['relerr']:.3e}",
        f" Timing (ms)     : Taylor={t['taylor']:.1f} | FD={t['fd']:.1f} | Speedup (FD/Taylor)={spd:.1f}×",
    ]
    print("\n".join(lines))


# ============================== quick run ==============================

if __name__ == "__main__":
    # Minimal smoke test (illustrative)
    S0, K, T = 100.0, 100.0, 1.0
    r, q = 0.05, 0.0
    T_knots = np.array([0.25, 0.5, 0.75, 1.0], dtype=float)
    sigma_slices = np.array([0.20, 0.22, 0.24, 0.25], dtype=float)
    w_slice_vals = (sigma_slices**2) * T_knots

    solver = BS_PDE_Taylor_DupireSlices(S0, K, T, r=r, q=q, M=101, N_base=120, S_max_mult=3.0)
    out = solver.solve_pde_taylor_w_slices_allseeds(
        S0_val=S0,
        T_knots=T_knots,
        w_slice_vals=w_slice_vals,
        verbose=True
    )

    print("\n--- Quick FD comparison ---")
    compare_dupire_taylor_vs_fd(
        solver_kwargs=dict(S0=S0, K=K, T=T, r=r, q=q, M=101, N_base=120, S_max_mult=3.0),
        S0_val=S0,
        T_knots=T_knots,
        w_slice_vals=w_slice_vals,
        verbose=True
    )
