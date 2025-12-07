import math
import time
import numpy as np
from numpy.linalg import cholesky, LinAlgError, eigh, inv, norm


try:
    from scipy.linalg import cho_factor, cho_solve, solve_triangular
except ImportError:
    print("[Warning] SciPy not found. Some linear-algebra routines will fall back to NumPy.")
    cho_factor, cho_solve, solve_triangular = None, None, None


# ============================================================
# Optional AAD library (aad_edge_pushing)
# ============================================================
try:
    from aad_edge_pushing.aad.core.var import ADVar
    from aad_edge_pushing.aad.core.tape import use_tape, global_tape
    from aad_edge_pushing.algo3.algo4_optimized import algo4_optimized

    try:
        from aad_edge_pushing.aad.ops.transcendental import exp, log, erf
    except ImportError:
        from aad_edge_pushing.aad.ops import exp, log

    AAD_AVAILABLE = True
    print("[Info] AAD library (aad_edge_pushing) loaded successfully.")
except ImportError:
    print("[Warning] AAD library (aad_edge_pushing) not found.")
    AAD_AVAILABLE = False



# ============================================================
# Shared numerical and linear-algebra utilities
# ============================================================
def _nearest_spd(A, eps=1e-12):
    B = 0.5 * (A + A.T)
    w, V = eigh(B)
    w_clip = np.clip(w, eps, None)
    return (V * w_clip) @ V.T


def _safe_cholesky(R, jitter=1e-12, max_tries=5):
    try:
        return cholesky(R)
    except LinAlgError:
        pass

    R_spd = _nearest_spd(R, eps=jitter)
    I = np.eye(R.shape[0])
    for k in range(max_tries):
        try:
            return cholesky(R_spd + (jitter * (10.0 ** k)) * I)
        except LinAlgError:
            continue
    return cholesky(R_spd + (jitter * (10.0 ** (max_tries + 2))) * I)


def _rinv_from_chol(L):
    if solve_triangular is not None:
        Linv = solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False)
        return Linv.T @ Linv
    else:
        Linv = inv(L)
        return Linv.T @ Linv


def _basket_lognormal_mm_baseline(S0, w, K, T, r, sigma, corr_like):
    
    S0 = np.asarray(S0, dtype=float)
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    R = np.asarray(corr_like, dtype=float)

    n = S0.size
    assert w.shape == (n,)
    assert sigma.shape == (n,)
    assert R.shape == (n, n)

    ES = S0 * np.exp(r * T)
    var_S = (S0 ** 2) * np.exp(2.0 * r * T) * (np.exp((sigma ** 2) * T) - 1.0)
    S0S0 = np.outer(S0, S0)
    sigsig = np.outer(sigma, sigma)
    cov_S = S0S0 * np.exp(2.0 * r * T) * (np.exp(sigsig * R * T) - 1.0)
    np.fill_diagonal(cov_S, var_S)

    m1 = float(w @ ES)
    s2 = float(w @ cov_S @ w)

    if s2 <= 0.0 or m1 <= 0.0:
        return max(m1 - K, 0.0)

    vB = math.log(1.0 + s2 / (m1 ** 2))
    muB = math.log(m1) - 0.5 * vB
    sqrtv = math.sqrt(vB)

    if K <= 0.0:
        return m1 - K

    d2 = (muB - math.log(K)) / sqrtv
    d1 = d2 + sqrtv

    from math import erf as math_erf, sqrt, exp
    Phi = lambda x: 0.5 * (1.0 + math_erf(x / sqrt(2.0)))
    return math.exp(muB + 0.5 * vB) * Phi(d1) - K * Phi(d2)


def softplus_scalar(x, alpha):

    ax = alpha * x
    if ax > 50.0:
        return x
    elif ax < -50.0:
        return 0.0
    else:
        return (math.log(1.0 + math.exp(ax))) / alpha


def payoff_smoothstep_ad(
    S_vars, w, K, T, r, sigma, drift, z_std,
    *, eps=0.01,
):

    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    n = len(S_vars)
    sqrtT = math.sqrt(T)

    B_ad = ADVar(0.0)
    for i in range(n):
        ci = math.exp(drift[i] + sigma[i] * sqrtT * float(z_std[i]))
        B_ad = B_ad + w[i] * S_vars[i] * ci

    x = B_ad - K
    xv = float(x.val)

    if xv <= 0.0:
        return B_ad * 0.0

    if xv >= eps:
        return x

    t = x / eps
    t2 = t * t
    return eps * (3.0 * t2 - 2.0 * t2 * t)


# ============================================================
# Method 1: LRM (score-function / likelihood-ratio method, S0-space)
# ============================================================
def basket_lrm_hessian_correlated(
    S0, w, K, T, r, sigma, corr, eps_steps,
    *,
    discount_at_end=True,
    use_baseline="none",
    sanity_print=False,
    tiny=1e-16,
):


    if not AAD_AVAILABLE:
        raise RuntimeError(
            "basket_lrm_hessian_correlated requires AAD + algo4_optimized. "
            "Please call it only when AAD_AVAILABLE=True."
        )

    t0 = time.perf_counter()
    S0 = np.asarray(S0, dtype=float)
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)
    eps_steps = np.asarray(eps_steps, dtype=float)

    n = S0.shape[0]
    n_paths, n_steps, _ = eps_steps.shape
    inv_sqrt_ns = 1.0 / math.sqrt(n_steps)

    sigma_eff = np.maximum(sigma, tiny)
    sqrtT = math.sqrt(T)
    drift_log = (r - 0.5 * (sigma_eff ** 2)) * T
    discount = math.exp(-r * T) if discount_at_end else 1.0

    S0_eff = np.maximum(S0, tiny)
    logS0 = np.log(S0_eff)
    Dinv = np.diag(1.0 / S0_eff)

    # ---------- Correlation matrix: Cholesky factor and inverse ----------
    L = _safe_cholesky(corr)
    R_used = L @ L.T

    Dsig_inv = np.diag(1.0 / sigma_eff)
    if cho_solve is not None and cho_factor is not None:
        try:
            cF = cho_factor(R_used, lower=True, check_finite=False)
            Rinv = cho_solve(cF, np.eye(n), check_finite=False)
        except Exception:
            Rinv = _rinv_from_chol(L)
    else:
        Rinv = _rinv_from_chol(L)

    # A = Gamma^{-1} = (1/T) * D_{1/sigma} * R^{-1} * D_{1/sigma}
    A = (1.0 / T) * (Dsig_inv @ Rinv @ Dsig_inv)
    m = logS0 + drift_log  # mean vector in log-space

    # ---------- Baseline (moment-matched lognormal) ----------
    if use_baseline == "mm":
        baseline = _basket_lognormal_mm_baseline(S0, w, K, T, r, sigma_eff, R_used)
    else:
        baseline = 0.0

    price_sum = 0.0
    H_sum = np.zeros((n, n), dtype=float)
    printed_once = False

    # Timing breakdown for profiling
    t_step1 = t_step2 = t_step3 = t_step4 = t_step5 = 0.0

    for p in range(n_paths):
        t_start = time.perf_counter()

        # ---------- Step 1: path generation (z_corr, ST, Xobs) ----------
        z_std = eps_steps[p].sum(axis=0) * inv_sqrt_ns
        z_corr = L @ z_std
        ST = S0_eff * np.exp(drift_log + sigma_eff * sqrtT * z_corr)
        Xobs = np.log(np.maximum(ST, tiny))
        t_after1 = time.perf_counter()

        # ---------- Step 2: basket payoff P ----------
        basket = float(np.dot(w, ST))
        P = max(basket - K, 0.0)
        price_sum += P
        t_after2 = time.perf_counter()

        # ---------- Step 3: analytic score g_ell ----------
        # g_ell = d ell / d S0
        d = Xobs - m
        u = A @ d
        g_ell = Dinv @ u  # shape (n,)
        t_after3 = time.perf_counter()

        # ---------- Step 4: build ell_ad via AAD and compute H_ell ----------
        global_tape.reset()
        with use_tape():
            S_vars = [ADVar(float(S0_eff[i])) for i in range(n)]

            m_ad = [log(S_vars[i]) + float(drift_log[i]) for i in range(n)]
            diff_ad = [float(Xobs[i]) - m_ad[i] for i in range(n)]

            ell_ad = ADVar(0.0)
            for i in range(n):
                for j in range(n):
                    ell_ad = ell_ad + (-0.5) * diff_ad[i] * float(A[i, j]) * diff_ad[j]

        H_ell = algo4_optimized(ell_ad, S_vars)
        t_after4 = time.perf_counter()

        # ---------- Step 5: LRM kernel and accumulation ----------
        # Kernel for Hessian of price: (P - baseline) * (H_ell + g_ell g_ell^T)
        weight = P - baseline
        kernel = H_ell + np.outer(g_ell, g_ell)
        H_sum += weight * kernel
        t_after5 = time.perf_counter()

        # Accumulate per-step timing
        t_step1 += (t_after1 - t_start)
        t_step2 += (t_after2 - t_after1)
        t_step3 += (t_after3 - t_after2)
        t_step4 += (t_after4 - t_after3)
        t_step5 += (t_after5 - t_after4)

        if sanity_print and (not printed_once):
            print("[LRM] (S0-space, correlated) using AAD + algo4_optimized to compute H_ell.")
            printed_once = True

    price = discount * (price_sum / n_paths)
    H = discount * (H_sum / n_paths)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"[Timing] Step 1 Generate random numbers, terminal asset prices, and log-observations: {t_step1 * 1000.0:.3f} ms")
    print(f"[Timing] Step 2 Compute the basket payoff for each path: {t_step2 * 1000.0:.3f} ms")
    print(f"[Timing] Step 3 Compute the first derivative (score) of the log-likelihood g_ell analytically: {t_step3 * 1000.0:.3f} ms")
    print(f"[Timing] Step 4 Use AAD to compute the second derivative (Hessian) of the log-likelihood H_ell: {t_step4 * 1000.0:.3f} ms")
    print(f"[Timing] Step 5 Form the LRM kernel from H_ell and g_ell and accumulate into the global Hessian estimate: {t_step5 * 1000.0:.3f} ms")

    return price, H, elapsed_ms


# ============================================================
# Method 2: Malliavin / IBP (differentiate w.r.t. X0 = log S0)
# ============================================================
def basket_malliavin_hessian_correlated(
    S0, w, K, T, r, sigma, corr, eps_steps,
    *,
    discount_at_end=True,
    use_baseline="none",
    tiny=1e-16,
    sanity_print=False,
):
    
    if not AAD_AVAILABLE:
        raise RuntimeError(
            "basket_malliavin_hessian_correlated requires AAD + algo4_optimized. "
            "Please call it only when AAD_AVAILABLE=True."
        )

    t0 = time.perf_counter()

    S0 = np.asarray(S0, dtype=float)
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)
    eps_steps = np.asarray(eps_steps, dtype=float)

    n = S0.size
    n_paths, n_steps, _ = eps_steps.shape
    inv_sqrt_ns = 1.0 / math.sqrt(n_steps)

    sigma_eff = np.maximum(sigma, tiny)
    sqrtT = math.sqrt(T)
    drift_log = (r - 0.5 * (sigma_eff ** 2)) * T
    discount = math.exp(-r * T) if discount_at_end else 1.0

    # Correlation matrix and inverse covariance in X-space
    L = _safe_cholesky(corr)
    R_used = L @ L.T
    Rinv = _rinv_from_chol(L)
    Dsig_inv = np.diag(1.0 / sigma_eff)
    A = (1.0 / T) * (Dsig_inv @ Rinv @ Dsig_inv)

    S0_eff = np.maximum(S0, tiny)
    logS0 = np.log(S0_eff)  # X0 = log S0
    m = logS0 + drift_log  # mean vector in X-space
    invS0 = 1.0 / S0_eff
    invS0_sq = invS0 * invS0

    if use_baseline == "mm":
        baseline = _basket_lognormal_mm_baseline(S0, w, K, T, r, sigma_eff, R_used)
    else:
        baseline = 0.0

    price_sum = 0.0
    H_S0_sum = np.zeros((n, n), dtype=float)
    printed_once = False

    t_step1 = t_step2 = t_step3 = t_step4 = t_step5 = 0.0

    for p in range(n_paths):
        t_start = time.perf_counter()

        # ---------- Step 1: path generation (z_corr, ST, Xobs) ----------
        z_std = eps_steps[p].sum(axis=0) * inv_sqrt_ns
        z_corr = L @ z_std
        ST = S0_eff * np.exp(drift_log + sigma_eff * sqrtT * z_corr)
        Xobs = np.log(np.maximum(ST, tiny))
        t_after1 = time.perf_counter()

        # ---------- Step 2: basket payoff P ----------
        basket = float(w @ ST)
        P = max(basket - K, 0.0)
        price_sum += P
        t_after2 = time.perf_counter()

        # ---------- Step 3: analytic gradient g_X0 ----------
        # d = Xobs - m, g_X0 = A d
        d = Xobs - m
        gX0 = A @ d
        t_after3 = time.perf_counter()

        # ---------- Step 4: construct ell_ad in X0-space and compute H_X0 via AAD ----------
        global_tape.reset()
        with use_tape():
            X0_vars = [ADVar(float(logS0[i])) for i in range(n)]

            m_ad = [X0_vars[i] + float(drift_log[i]) for i in range(n)]
            diff_ad = [float(Xobs[i]) - m_ad[i] for i in range(n)]

            ell_ad = ADVar(0.0)
            for i in range(n):
                for j in range(n):
                    ell_ad = ell_ad + (-0.5) * diff_ad[i] * float(A[i, j]) * diff_ad[j]

        H_X0 = algo4_optimized(ell_ad, X0_vars)
        t_after4 = time.perf_counter()

        # ---------- Step 5: Malliavin / IBP kernel in X0-space + chain rule to S0 ----------
        weight = P - baseline

        # Kernel in X0-space: H_X0 + gX0 gX0^T
        kernel_X0 = H_X0 + np.outer(gX0, gX0)
        H_path_X0 = weight * kernel_X0

        # Chain rule from X0 = log S0 back to S0:
        # H_S0 = D^{-1} H_path_X0 D^{-1} - diag(weight * gX0 * invS0^2)
        H_term1 = (invS0[:, None] * H_path_X0) * invS0[None, :]
        H_term2 = np.diag(weight * gX0 * invS0_sq)
        H_S0 = H_term1 - H_term2

        H_S0_sum += H_S0
        t_after5 = time.perf_counter()

        # Accumulate per-step timing
        t_step1 += (t_after1 - t_start)
        t_step2 += (t_after2 - t_after1)
        t_step3 += (t_after3 - t_after2)
        t_step4 += (t_after4 - t_after3)
        t_step5 += (t_after5 - t_after4)

        if sanity_print and (not printed_once):
            print("[Malliavin/IBP] (X0-space) using AAD + algo4_optimized to compute H_X0, then mapping to S0-space via the chain rule.")
            printed_once = True

    price = discount * (price_sum / n_paths)
    H = discount * (H_S0_sum / n_paths)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Print timing for steps 1–5 (Malliavin estimator, in ms)
    print(f"[Timing] Step 1 Generate random numbers, terminal asset prices, and log-observations: {t_step1 * 1000.0:.3f} ms")
    print(f"[Timing] Step 2 Compute the basket payoff for each path: {t_step2 * 1000.0:.3f} ms")
    print(f"[Timing] Step 3 Compute the gradient of the log-likelihood with respect to X0 (g_X0) analytically: {t_step3 * 1000.0:.3f} ms")
    print(f"[Timing] Step 4 Use AAD to compute the Hessian of the log-likelihood with respect to X0 (H_X0): {t_step4 * 1000.0:.3f} ms")
    print(f"[Timing] Step 5 Form the Malliavin/IBP kernel in X0-space, transform it to S0-space via the chain rule, and accumulate into the Hessian estimate: {t_step5 * 1000.0:.3f} ms")

    return price, H, elapsed_ms


# ============================================================
# Method 3: Kernel convolution (AAD-based Gaussian smoothing)
# ============================================================
def basket_kernel_hessian(
    S0, w, K, T, r, sigma, eps_steps,
    *,
    eps_kernel=0.05,
    corr=None,
    chol_corr=None,
    sanity_print=False,
    tiny=1e-16,
):

    if not AAD_AVAILABLE:
        print("[Error] AAD library not available. Cannot run Kernel Convolution method.")
        n = len(S0)
        return 0.0, np.zeros((n, n)), 0.0

    t_start = time.perf_counter()

    S0 = np.asarray(S0, dtype=float)
    w = np.asarray(w, dtype=float)
    sigma_raw = np.asarray(sigma, dtype=float)
    eps_steps = np.asarray(eps_steps, dtype=float)

    n = len(S0)
    n_paths = eps_steps.shape[0]
    n_steps = eps_steps.shape[1]
    inv_sqrt_ns = 1.0 / math.sqrt(n_steps)

    sigma_eff = np.maximum(sigma_raw, tiny)
    drift_log = (r - 0.5 * sigma_eff ** 2) * T
    discount = math.exp(-r * T)
    sqrtT = math.sqrt(T)

    # Correlation structure (pre-computed Cholesky can be passed in)
    if chol_corr is not None:
        L = np.asarray(chol_corr, dtype=float)
    elif corr is not None:
        L = np.linalg.cholesky(np.asarray(corr, dtype=float))
    else:
        L = np.eye(n, dtype=float)

    price_sum = 0.0
    H_sum = np.zeros((n, n))

    # --- 2. Helper: analytic Gaussian smoothing of a call payoff ---
    INV_SQRT_2PI = 0.3989422804014327

    def analytic_call_smooth(basket_val, strike, eps):
        x = basket_val - strike

        if eps < 1e-14:
            return x if (x.val > 0) else ADVar(0.0)

        d = x / eps

        pdf = exp(-0.5 * d * d) * INV_SQRT_2PI
        cdf = 0.5 * (1.0 + erf(d * 0.7071067811865475))

        return x * cdf + eps * pdf

    # Timing breakdown
    t1 = t2 = t3 = t4 = t5 = 0.0
    printed_once = False

    for p in range(n_paths):
        t0 = time.perf_counter()

        # Step 1: generate correlated shocks
        z_std_ind = eps_steps[p].sum(axis=0) * inv_sqrt_ns
        z_corr = L @ z_std_ind
        t1p = time.perf_counter()

        # Step 2: (placeholder – currently no extra work here)
        t2p = time.perf_counter()

        # Step 3: build AAD graph for ST and smoothed payoff
        global_tape.reset()
        with use_tape():
            S_vars = [ADVar(float(S0[i])) for i in range(n)]

            ST_vec = []
            for i in range(n):
                log_ST_i = log(S_vars[i]) + (drift_log[i] + sigma_eff[i] * sqrtT * z_corr[i])
                ST_vec.append(exp(log_ST_i))

            basket_ad = sum(w[i] * ST_vec[i] for i in range(n))
            payoff_ad = analytic_call_smooth(basket_ad, K, eps_kernel)
            price_ad = discount * payoff_ad
            t4p = time.perf_counter()

        # Step 4: reverse-mode Hessian via algo4_optimized
        H_path = algo4_optimized(price_ad, S_vars)
        t5p = time.perf_counter()

        # Step 5: accumulate into MC estimator
        price_sum += float(price_ad.val)
        H_sum += H_path
        t6p = time.perf_counter()

        # Timing accumulation
        t1 += (t1p - t0)
        t2 += (t2p - t1p)
        t3 += (t4p - t2p)
        t4 += (t5p - t4p)
        t5 += (t6p - t5p)

        if sanity_print and (not printed_once):
            print("Using analytic Gaussian smoothing and AAD to compute the Hessian.")
            printed_once = True

    price = price_sum / n_paths
    H = H_sum / n_paths

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    print(f"[Timing] Step 1 Asset shocks (MC): {t1 * 1000:.3f} ms")
    print(f"[Timing] Step 2 (currently unused placeholder): {t2 * 1000:.3f} ms")
    print(f"[Timing] Step 3 AAD graph build + smoothed payoff: {t3 * 1000:.3f} ms")
    print(f"[Timing] Step 4 algo4_optimized Hessian: {t4 * 1000:.3f} ms")
    print(f"[Timing] Step 5 Accumulate into global Hessian: {t5 * 1000:.3f} ms")

    return price, H, elapsed_ms


# ============================================================
# Method 4: Unified finite-difference (FD) benchmark
# ============================================================
def basket_fd_hessian(
    S0, w, K, T, r, sigma, L, eps_steps,
    alpha=20.0,
    h_rel=1e-4,
    tiny=1e-16,
):

    t_start = time.perf_counter()

    S0 = np.array(S0, dtype=float)
    w = np.array(w, dtype=float)
    sigma = np.array(sigma, dtype=float)
    sigma_eff = np.maximum(sigma, tiny)

    n = len(S0)
    n_paths, n_steps = eps_steps.shape[0], eps_steps.shape[1]

    inv_sqrt_ns = 1.0 / math.sqrt(n_steps)
    sqrtT = math.sqrt(T)
    discount = math.exp(-r * T)
    h = h_rel * np.maximum(1.0, np.abs(S0))

    # Timing breakdown
    t_step1 = t_step2 = t_step3 = t_step4 = t_step5 = 0.0

    def price_fn(S_vec):
        nonlocal t_step1, t_step2, t_step3, t_step4

        total = 0.0
        drift_log = (r - 0.5 * (sigma_eff ** 2.0)) * T
        S_vec_eff = np.maximum(S_vec, tiny)

        for p in range(n_paths):
            t0_p = time.perf_counter()

            # ---------- Step 1: generate z_std, z_corr ----------
            z_std = eps_steps[p].sum(axis=0) * inv_sqrt_ns
            z_corr = L @ z_std
            t1_p = time.perf_counter()

            # ---------- Step 2: generate terminal vector ST_vec ----------
            ST_vec = S_vec_eff * np.exp(drift_log + sigma_eff * sqrtT * z_corr)
            t2_p = time.perf_counter()

            # ---------- Step 3: compute basket and Softplus payoff ----------
            basket = float(np.dot(w, ST_vec))
            payoff = softplus_scalar(basket - K, alpha=alpha)
            t3_p = time.perf_counter()

            # ---------- Step 4: discount and accumulate ----------
            total += discount * payoff
            t4_p = time.perf_counter()

            t_step1 += (t1_p - t0_p)
            t_step2 += (t2_p - t1_p)
            t_step3 += (t3_p - t2_p)
            t_step4 += (t4_p - t3_p)

        return total / n_paths

    price = price_fn(S0)
    H = np.zeros((n, n))

    # ---------- Step 5: assemble full FD Hessian (diagonal + off-diagonal) ----------
    t5_start = time.perf_counter()

    # Diagonal terms
    for i in range(n):
        S_up = S0.copy(); S_up[i] += h[i]
        S_dn = S0.copy(); S_dn[i] -= h[i]
        f_up = price_fn(S_up)
        f_dn = price_fn(S_dn)
        H[i, i] = (f_up - 2 * price + f_dn) / (h[i] ** 2)

    # Off-diagonal cross terms
    for i in range(n):
        for j in range(i + 1, n):
            S_pp = S0.copy(); S_pp[i] += h[i]; S_pp[j] += h[j]
            S_pm = S0.copy(); S_pm[i] += h[i]; S_pm[j] -= h[j]
            S_mp = S0.copy(); S_mp[i] -= h[i]; S_mp[j] += h[j]
            S_mm = S0.copy(); S_mm[i] -= h[i]; S_mm[j] -= h[j]

            f_pp = price_fn(S_pp)
            f_pm = price_fn(S_pm)
            f_mp = price_fn(S_mp)
            f_mm = price_fn(S_mm)

            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            H[i, j] = H_ij
            H[j, i] = H_ij

    t5_end = time.perf_counter()
    t_step5 += (t5_end - t5_start)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    n_evals = 1 + 2 * n + 4 * (n * (n - 1) // 2)

    print(f"[Timing] Step 1 For each path, generate independent normals and apply the Cholesky factor L to obtain correlated shocks: {t_step1 * 1000.0:.3f} ms")
    print(f"[Timing] Step 2 For each path, generate the terminal asset vector ST_vec from the current S0 input and the correlated shocks: {t_step2 * 1000.0:.3f} ms")
    print(f"[Timing] Step 3 For each path, compute the basket value and the smoothed Softplus payoff: {t_step3 * 1000.0:.3f} ms")
    print(f"[Timing] Step 4 Discount the payoff and accumulate it into the Monte Carlo price estimate: {t_step4 * 1000.0:.3f} ms")
    print(f"[Timing] Step 5 Assemble the full finite-difference Hessian with respect to S0 by repeatedly evaluating the pricing function at shifted points: {t_step5 * 1000.0:.3f} ms")

    return price, H, elapsed_ms, n_evals


n_assets_global = 15

if not AAD_AVAILABLE:
    ADVar = lambda x: None

    def use_tape():
        def _wrapper(fn):
            return fn
        return _wrapper

    class GlobalTapeStub:
        def reset(self):
            pass

    global_tape = GlobalTapeStub()
    algo4_optimized = lambda *args, **kwargs: np.zeros((n_assets_global, n_assets_global))


# ============================================================
# Executable example: compare LRM vs Malliavin vs Kernel vs FD
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print(" Running LRM vs Malliavin/IBP vs Kernel(AAD) vs FD (benchmark)")
    print("=" * 60)

    # --- 1. Market and option parameters ---
    n_assets = 15
    n_assets_global = n_assets

    S0 = np.random.uniform(80.0, 120.0, size=n_assets)
    w = np.random.dirichlet(np.ones(n_assets))
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = np.random.uniform(0.15, 0.35, size=n_assets)

    A = np.random.normal(size=(n_assets, n_assets))
    C = A @ A.T
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)

    # --- 2. Monte Carlo parameters ---
    n_paths = 1000
    n_steps = 1

    # --- 3. Generate random numbers ---
    np.random.seed(42)
    eps_steps = np.random.randn(n_paths, n_steps, n_assets)

    # --- 4. Precompute Cholesky (shared across methods) ---
    L = _safe_cholesky(corr)

    # --- 5. Run LRM estimator (with moment-matching baseline) ---
    print("\n" + "-" * 20 + " 1. LRM (S0-space) " + "-" * 20)
    try:
        price_lrm, H_lrm, elapsed_lrm = basket_lrm_hessian_correlated(
            S0, w, K, T, r, sigma, corr, eps_steps,
            use_baseline="mm",
            sanity_print=True,
        )
        print(f"N_paths: {n_paths}")
        print(f"Total time: {elapsed_lrm:.2f} ms")
        print(f"Estimated price: {price_lrm:.6f}")
        print("Estimated Hessian (H_LRM):")
        with np.printoptions(precision=6, suppress=True):
            print(H_lrm)

    except Exception as e:
        print(f"\nError when running LRM: {e}")
        import traceback

        traceback.print_exc()
        price_lrm, H_lrm, elapsed_lrm = 0, np.zeros((n_assets, n_assets)), -1

    # --- 6. Run Malliavin/IBP estimator (with moment-matching baseline) ---
    print("\n" + "-" * 20 + " 2. Malliavin (X0-space) " + "-" * 20)
    try:
        price_mall, H_mall, elapsed_mall = basket_malliavin_hessian_correlated(
            S0, w, K, T, r, sigma, corr, eps_steps,
            use_baseline="mm",
            sanity_print=True,
        )
        print(f"N_paths: {n_paths}")
        print(f"Total time: {elapsed_mall:.2f} ms")
        print(f"Estimated price: {price_mall:.6f}")
        print("Estimated Hessian (H_Malliavin):")
        with np.printoptions(precision=6, suppress=True):
            print(H_mall)

    except Exception as e:
        print(f"\nError when running Malliavin/IBP: {e}")
        import traceback

        traceback.print_exc()
        price_mall, H_mall, elapsed_mall = 0, np.zeros((n_assets, n_assets)), -1

    # --- 7. Run FD estimator (as benchmark) ---
    print("\n" + "-" * 20 + " 3. FD (benchmark) " + "-" * 20)
    fd_n_evals = 1 + 2 * n_assets + 4 * (n_assets * (n_assets - 1) // 2)
    print(f"[Note] FD is relatively slow: it requires {fd_n_evals} pricing evaluations...")

    fd_alpha = 25

    try:
        fd_h_rel = 1e-4

        price_fd, H_fd, elapsed_fd, n_evals = basket_fd_hessian(
            S0, w, K, T, r, sigma, L, eps_steps,
            alpha=fd_alpha,
            h_rel=fd_h_rel,
        )

        print(f"N_paths: {n_paths}")
        print(f"FD calls: {n_evals}")
        print(f"FD Softplus alpha: {fd_alpha}")
        print(f"Total time: {elapsed_fd:.2f} ms")
        print(f"Estimated price (Softplus): {price_fd:.6f}")
        print("Estimated Hessian (H_FD):")
        with np.printoptions(precision=6, suppress=True):
            print(H_fd)

    except Exception as e:
        print(f"\nError when running FD (benchmark): {e}")
        import traceback

        traceback.print_exc()
        price_fd, H_fd, elapsed_fd = 0, np.zeros((n_assets, n_assets)), -1

    # --- 8. Run Kernel (AAD convolution) estimator ---
    print("\n" + "-" * 20 + " 4. Kernel (AAD) [Convolution] " + "-" * 20)
    if not AAD_AVAILABLE:
        print("AAD library not installed; skipping Kernel (AAD Convolution) method.")
        price_kern, H_kern, elapsed_kern = 0, np.zeros((n_assets, n_assets)), -1
    else:
        try:
            eps_kernel = 0.05
            price_kern, H_kern, elapsed_kern = basket_kernel_hessian(
                S0, w, K, T, r, sigma, eps_steps,
                eps_kernel=eps_kernel,
                corr=corr,
                sanity_print=False,
            )
            print(f"N_paths: {n_paths}")
            print(f"Kernel epsilon (smoothing): {eps_kernel}")
            print(f"Total time: {elapsed_kern:.2f} ms")
            print(f"Estimated price (Kernel-Convolution): {price_kern:.6f}")
            print("Estimated Hessian (H_Kernel):")
            with np.printoptions(precision=6, suppress=True):
                print(H_kern)
        except Exception as e:
            print(f"\nError when running Kernel (AAD Convolution): {e}")
            import traceback

            traceback.print_exc()
            price_kern, H_kern, elapsed_kern = 0, np.zeros((n_assets, n_assets)), -1

    # --- 9. Final comparison ---
    print("\n" + "=" * 60)
    print("--- Final comparison (Hessian errors vs FD benchmark) ---")
    print("=" * 60)

    if elapsed_fd <= 0:
        print("FD benchmark failed; cannot compare methods.")
    else:
        # LRM vs FD
        if elapsed_lrm >= 0:
            diff_lrm = H_lrm - H_fd
            norm_lrm = norm(diff_lrm)
            print(f"H_LRM vs H_FD (L2 norm): {norm_lrm:.6e}")
            print(f"  (LRM speed: {elapsed_lrm:.2f} ms vs FD: {elapsed_fd:.2f} ms. ~{elapsed_fd / elapsed_lrm:.1f}x faster)")

        # Malliavin vs FD
        if elapsed_mall >= 0:
            diff_mall = H_mall - H_fd
            norm_mall = norm(diff_mall)
            print(f"H_Malliavin vs H_FD (L2 norm): {norm_mall:.6e}")
            print(f"  (Malliavin speed: {elapsed_mall:.2f} ms vs FD: {elapsed_fd:.2f} ms. ~{elapsed_fd / elapsed_mall:.1f}x faster)")

        # Kernel vs FD
        if elapsed_kern >= 0:
            diff_kern = H_kern - H_fd
            norm_kern = norm(diff_kern)
            print(f"H_Kernel (Softplus) vs H_FD (Softplus) (L2 norm): {norm_kern:.6e}")
            print(f"  (Kernel speed: {elapsed_kern:.2f} ms vs FD: {elapsed_fd:.2f} ms. ~{elapsed_fd / elapsed_kern:.1f}x faster)")
