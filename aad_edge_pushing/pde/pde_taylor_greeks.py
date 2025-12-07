# pde_taylor_greeks.py - CLEAN PDE+Taylor ONLY
"""
PDE + Taylor Expansion Greeks Calculator (single volatility seed)

What this file does:
- Price a European call via Black–Scholes PDE (S-space) with Crank–Nicolson.
- Treat volatility sigma as an AD seed and run a single Taylor backprop to obtain:
    * price V
    * vega  dV/dsigma
    * volga d2V/dsigma^2
    * vanna d/dsigma (Delta)  [via Taylor on an AD Delta]
- Delta and Gamma (w.r.t. S) are read from the S-grid numerically for stability.
- Validation helper compares PDE+Taylor to BSM closed-form greeks.

Notes:
- Coefficients are built ONCE with ADVar(sigma) and reused across CN steps
  (the AD dependency is captured inside the coefficients).
- Boundaries are numeric (wrapped as ADVar constants), which is fine for
  sigma-seeded greeks. If you need precise dV/dK from payoff, replace the
  numeric terminal payoff with an AD-smooth payoff (e.g., softplus) so K
  participates in the tape. That is out of scope here by design.
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad.core.var import ADVar
from aad.core.tape import global_tape
from aad.core.taylor_backprop import taylor_backpropagate


def _taylor_grad_hess_of(target_advar, seeds_list):
    """
    Helper: always backprop wrt `seeds_list` for the given target.
    Some implementations accept targets=[...]; fall back if not.
    """
    try:
        return taylor_backpropagate(seeds_list, targets=[target_advar])
    except TypeError:
        target_advar.is_output = True
        return taylor_backpropagate(seeds_list)


class BS_PDE_Taylor:
    """
    Black–Scholes PDE solver in S-space with Taylor backprop on sigma.

    Model:
        dV/dt + 0.5*sigma^2*S^2 * d2V/dS2 + r*S*dV/dS - r*V = 0

    Scheme:
        - Crank–Nicolson with weight phi in [0,1], default 0.5 (classical CN)
        - Uniform S-grid in [0, S_max_mult*K]
        - Numeric (float) boundaries plugged into the RHS each step

    Greeks:
        - seeds: sigma only
        - price   V       : AD scalar (via interp at S0)
        - vega    dV/dσ   : Taylor grad wrt sigma
        - volga   d2V/dσ2 : Taylor Hessian wrt sigma
        - vanna   d/dσ(Δ) : Taylor grad of an AD Delta constructed on the grid
        - delta, gamma    : numeric (central differences on S-grid)
    """

    def __init__(self,
                 S0: float, K: float, T: float,
                 r,                    # float or callable(t)->float
                 q=0.0,                # float or callable(t)->float
                 M: int = 101, N_base: int = 100,
                 S_max_mult: float = 3.0,
                 phi: float = 0.5):
        self.S0 = float(S0)
        self.K  = float(K)
        self.T  = float(T)
        self.M  = int(M)
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

        # uniform S-grid
        S_min, S_max = 0.0, S_max_mult * self.K
        self.S_grid = np.linspace(S_min, S_max, self.M)
        self.dS = self.S_grid[1] - self.S_grid[0]

    # ---------------- terminal & boundary conditions (numeric) ----------------

    def _terminal_condition(self) -> np.ndarray:
        """European call payoff on the S-grid at t = T (numeric array)."""
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_condition_left(self, t: float) -> float:
        """As S -> 0, European call ~ 0."""
        return 0.0

    def _boundary_condition_right(self, t: float) -> float:
        """
        As S -> S_max, call ~ S_max * e^{-q (T - t)} - K e^{-r (T - t)} (numeric).
        """
        tau = self.T - t
        r_now = self.r(t)
        q_now = self.q(t)
        S_max = self.S_grid[-1]
        return S_max * np.exp(-q_now * tau) - self.K * np.exp(-r_now * tau)

    # ---------------- stability-oriented time grid (optional) ----------------

    def compute_adaptive_timesteps(self, sigma: float) -> Tuple[np.ndarray, int]:
        """
        Choose N such that alpha_max * dt <= ~0.5 at S_max (simple heuristic).
        Falls back to N_base if sigma is tiny; caps N to avoid huge tapes.
        """
        S_max = self.S_grid[-1]
        dS = self.dS
        alpha_max = (sigma**2 * S_max**2 / 2.0) / (dS**2)
        dt_stable = 0.5 / alpha_max if alpha_max > 1e-10 else self.T / self.N_base
        N = max(int(np.ceil(self.T / dt_stable)), self.N_base)

        N_max = 200
        if N > N_max:
            N = N_max

        t_grid = np.linspace(0, self.T, N + 1)
        return t_grid, N

    # ---------------- CN coefficient assembly (with AD sigma) ----------------

    def build_tridiagonal_cn(self, sigma_var: ADVar, dt: ADVar):
        """
        Build CN tri-diagonals once with ADVar(sigma); reuse at each step.
        r is frozen at r(0) here; this is fine when the only seed is sigma.
        """
        n = self.M - 2
        dS = self.dS
        dS_sq = ADVar(dS**2, requires_grad=False)
        dS_2  = ADVar(2.0 * dS, requires_grad=False)
        phi   = ADVar(self.phi, requires_grad=False)
        one_minus_phi = ADVar(1.0 - self.phi, requires_grad=False)

        r0 = self.r(0.0) if callable(self.r) else float(self.r)
        r0 = ADVar(r0, requires_grad=False)

        a_L, b_L, c_L, a_R, b_R, c_R = [], [], [], [], [], []
        for i in range(n):
            S_i = self.S_grid[i + 1]
            S_i_var = ADVar(S_i, requires_grad=False)

            alpha_i = (sigma_var * sigma_var * S_i_var * S_i_var / ADVar(2.0)) / dS_sq
            beta_i  = (r0 * S_i_var) / dS_2
            gamma_i = ADVar(-1.0, requires_grad=False) * r0

            l_i = alpha_i - beta_i
            c_i = ADVar(-2.0) * alpha_i + gamma_i
            u_i = alpha_i + beta_i

            a_L.append(ADVar(0.0) if i == 0 else ADVar(-1.0) * phi * dt * l_i)
            b_L.append(ADVar(1.0) - phi * dt * c_i)
            c_L.append(ADVar(0.0) if i == n - 1 else ADVar(-1.0) * phi * dt * u_i)

            a_R.append(ADVar(0.0) if i == 0 else one_minus_phi * dt * l_i)
            b_R.append(ADVar(1.0) + one_minus_phi * dt * c_i)
            c_R.append(ADVar(0.0) if i == n - 1 else one_minus_phi * dt * u_i)
        return a_L, b_L, c_L, a_R, b_R, c_R

    # ---------------- Thomas solver & one CN step ----------------

    def tridiag_solve(self, a, b, c, d):
        """Standard Thomas algorithm with ADVar entries."""
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

    def cn_step(self, V, a_L, b_L, c_L, a_R, b_R, c_R, t_current):
        """
        Assemble RHS using numeric boundaries (wrapped as AD constants),
        then solve with the prebuilt tri-diagonals.
        """
        n = self.M - 2
        rhs = [None] * n

        for i in range(n):
            if i == 0:
                V_left = ADVar(self._boundary_condition_left(t_current), requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i + 1] - a_R[i] * V_left
            elif i == n - 1:
                V_right = ADVar(self._boundary_condition_right(t_current), requires_grad=False)
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                rhs[i] = a_R[i] * V[i - 1] + b_R[i] * V[i] + c_R[i] * V[i + 1]

        return self.tridiag_solve(a_L, b_L, c_L, rhs)

    # ---------------- main API: single sigma seed ----------------

    def solve_pde_taylor_single_var(self, S0_val: float, sigma_val: float,
                                    verbose: bool = False) -> Dict:
        """
        Solve PDE with a single AD seed sigma, then run Taylor backprop to get:
          price, vega, volga; numeric delta/gamma; and vanna = d/dsigma(Delta).
        """
        t_start = time.perf_counter()
        global_tape.reset()

        sigma_var = ADVar(sigma_val, requires_grad=True, name="sigma")
        self.S0 = float(S0_val)

        # time grid (stability heuristic)
        t_grid, N = self.compute_adaptive_timesteps(sigma_val)
        if verbose:
            print(f"Grid: M={self.M}, N={N}, dt={(t_grid[1]-t_grid[0]):.6f}, dS={self.dS:.6f}")

        dt_val = t_grid[1] - t_grid[0]
        dt = ADVar(dt_val, requires_grad=False)

        # build coefficients ONCE (captures AD dep on sigma)
        t0 = time.perf_counter()
        a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_cn(sigma_var, dt)
        build_ms = (time.perf_counter() - t0) * 1000.0

        # terminal condition (numeric -> AD constants)
        V_terminal = self._terminal_condition()
        V = [ADVar(float(v), requires_grad=False) for v in V_terminal[1:-1]]

        # CN stepping from T -> 0 using the same matrices (sigma AD-aware)
        t0 = time.perf_counter()
        for n in range(N):
            t_cur = t_grid[n + 1]  # going backward; boundary uses numeric t
            V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_cur)
        step_ms = (time.perf_counter() - t0) * 1000.0

        # linear interpolation at S0 for price (AD expression)
        S_interior = self.S_grid[1:-1]
        idx = np.searchsorted(S_interior, self.S0)
        idx = max(1, min(idx, len(V) - 1))
        S1, S2 = S_interior[idx - 1], S_interior[idx]
        w = (self.S0 - S1) / (S2 - S1)
        price_var = V[idx - 1] * ADVar(1.0 - w, requires_grad=False) + V[idx] * ADVar(w, requires_grad=False)

        # Taylor backprop for vega/volga
        t0 = time.perf_counter()
        grad_p, H_p = taylor_backpropagate([sigma_var], target=price_var)
        back_ms = (time.perf_counter() - t0) * 1000.0

        price = float(price_var.val)
        vega  = float(grad_p[0])
        volga = float(H_p[0, 0])

        # numeric Delta/Gamma on the grid (stable)
        V_vals = np.array([v.val for v in V])
        delta = self._compute_delta_on_grid(V_vals, self.S0)
        gamma = self._compute_gamma_on_grid(V_vals, self.S0)

        # vanna = d/dsigma(Delta): construct AD Delta on V (ADVars) and backprop wrt sigma
        idx_c = np.searchsorted(S_interior, self.S0)
        idx_c = max(1, min(idx_c, len(V) - 2))
        dS = self.dS
        delta_var = (V[idx_c + 1] - V[idx_c - 1]) / ADVar(2.0 * dS, requires_grad=False)
        grad_d, _ = taylor_backpropagate([sigma_var], target=delta_var)
        vanna = float(grad_d[0])

        total_ms = (time.perf_counter() - t_start) * 1000.0
        if verbose:
            print(f"Build: {build_ms:.2f} ms | Stepping: {step_ms:.2f} ms | Taylor: {back_ms:.2f} ms | Total: {total_ms:.2f} ms")

        return {
            "price": price,
            "delta": float(delta),
            "gamma": float(gamma),
            "vega":  vega,
            "volga": volga,
            "vanna": vanna,
            "time_ms": total_ms,
            "tape_nodes": len(global_tape.nodes),
            "method": "pde+taylor_single_sigma"
        }

    # ---------------- finite-difference reads on S-grid ----------------

    def _compute_delta_on_grid(self, V_grid: np.ndarray, S0: float) -> float:
        """Central difference Delta at S0 on the interior grid."""
        idx = np.searchsorted(self.S_grid[1:-1], S0)
        idx = max(1, min(idx, len(V_grid) - 2))
        return (V_grid[idx + 1] - V_grid[idx - 1]) / (2.0 * self.dS)

    def _compute_gamma_on_grid(self, V_grid: np.ndarray, S0: float) -> float:
        """Second central difference Gamma at S0 on the interior grid."""
        idx = np.searchsorted(self.S_grid[1:-1], S0)
        idx = max(1, min(idx, len(V_grid) - 2))
        return (V_grid[idx + 1] - 2.0 * V_grid[idx] + V_grid[idx - 1]) / (self.dS ** 2)


# =============================================================================
# Validation against Black–Scholes closed-form (single sigma seed)
# =============================================================================

def validate_against_bsm(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
                         M=101, N_base=100, verbose=True):
    """
    Compare PDE+Taylor results to BSM analytical greeks under constant sigma.
    """
    from scipy.stats import norm

    print("=" * 72)
    print("VALIDATION: PDE+Taylor (single sigma seed) vs BSM analytical")
    print("=" * 72)

    def bsm_greeks(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        vega  = S * norm.pdf(d1) * np.sqrt(T)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        volga = vega * d1 * d2 / sigma
        vanna = -norm.pdf(d1) * d2 / sigma
        return dict(price=price, delta=delta, gamma=gamma, vega=vega, volga=volga, vanna=vanna)

    bsm = bsm_greeks(S0, K, T, r, sigma)

    solver = BS_PDE_Taylor(S0, K, T, r, M=M, N_base=N_base)
    pde = solver.solve_pde_taylor_single_var(S0, sigma, verbose=verbose)

    print(f"\n{'Greek':<10} {'BSM':>15} {'PDE+Taylor':>15} {'Abs.Err':>15} {'Rel.Err%':>10}")
    print("-" * 72)
    for greek in ["price", "delta", "gamma", "vega", "volga", "vanna"]:
        b = float(bsm[greek])
        a = float(pde[greek])
        err = abs(a - b)
        rel = (err / abs(b) * 100.0) if b != 0 else 0.0
        print(f"{greek:<10} {b:>15.8f} {a:>15.8f} {err:>15.8f} {rel:>10.4f}")

    print(f"\nComputation time: {pde['time_ms']:.2f} ms | Tape nodes: {pde['tape_nodes']:,}")
    return bsm, pde


if __name__ == "__main__":
    # Minimal self-check
    validate_against_bsm(verbose=True)
