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
# AAD (aad_edge_pushing) imports
# ============================================================
try:
    from aad_edge_pushing.aad.core.var import ADVar
    from aad_edge_pushing.aad.core.tape import use_tape, global_tape
    from aad_edge_pushing.algo3.algo4_optimized import algo4_optimized

    try:
        from aad_edge_pushing.aad.ops.transcendental import exp, log
    except ImportError:
        from aad_edge_pushing.aad.ops import exp, log

    AAD_AVAILABLE = True
    print("[Info] AAD library (aad_edge_pushing) loaded successfully. CE/Vibrato methods are enabled.")
except ImportError:
    print("[Warning] AAD library (aad_edge_pushing) not found. CE/Vibrato (AAD) methods will be skipped.")
    AAD_AVAILABLE = False


# ============================================================
# Softplus for both ADVar and float
# ============================================================
def softplus_ad(x, alpha):
    """
    Stable Softplus that works for both ADVar and float:
        softplus(x) = (1/alpha) * log(1 + exp(alpha * x))
    with saturation for large |alpha * x|.
    """
    ax = alpha * x

    # ADVar branch: use ax.val to decide saturation, but keep AD structure
    if AAD_AVAILABLE and isinstance(x, ADVar):
        v = ax.val
        if v > 30.0:
            # Large positive: softplus(x) ≈ x
            return x
        elif v < -30.0:
            # Large negative: softplus(x) ≈ 0
            return x * 0.0
        else:
            return (1.0 / alpha) * log(1.0 + exp(ax))
    else:
        # Pure float branch
        ax_float = float(ax)
        if ax_float > 30.0:
            return float(x)
        elif ax_float < -30.0:
            return 0.0
        else:
            return (1.0 / alpha) * math.log1p(math.exp(ax_float))


# ============================================================
# Small numerical helpers (smooth sign / erf approximation)
# ============================================================
EPS_SMOOTH_SQRT = 1e-8
CLAMP_ARG = 40.0  # NOTE: currently unused; kept for potential future clamping logic.


def _sqrt(x):
    """Thin wrapper that allows ADVar ** 0.5."""
    return x ** 0.5


def _sign_smooth(x, eps=EPS_SMOOTH_SQRT):
    """
    Smooth approximation of sign(x) using
        x / (sqrt(x^2 + eps) + eps)
    to avoid discontinuity at 0 in AD context.
    """
    return x / (_sqrt(x * x + eps) + eps)


def _erf_approx(x, eps=EPS_SMOOTH_SQRT):
    """
    Smooth approximation of erf(x) that is AD-friendly.
    Uses a classical polynomial approximation with a
    smooth sign and AAD 'exp'.
    """
    ax = _sqrt(x * x + eps)
    sgn = _sign_smooth(x, eps)
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    t = 1.0 / (1.0 + p * ax)
    poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
    y = 1.0 - poly * exp(-(ax * ax))
    return sgn * y


def normal_cdf(u):
    """
    Normal CDF Φ(u) implemented via the AD-friendly erf approximation.
    """
    inv_sqrt2 = 1.0 / (2.0 ** 0.5)
    return 0.5 * (1.0 + _erf_approx(u * inv_sqrt2))


# ============================================================
# Misc helpers
# ============================================================
def _choose_j_index(w, sigma):
    """
    Choose the conditioning asset index j in CE-on-Asset:
    pick the asset with the largest |w_i| * sigma_i.
    """
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    mask = (np.abs(w) > 0)
    if not np.any(mask):
        return 0
    score = np.abs(w) * np.maximum(sigma, 1e-16)
    return int(np.argmax(score))


def _nearest_spd(A, eps=1e-12):
    """
    Project a real symmetric matrix A to the nearest SPD matrix
    via eigenvalue clipping (simplified Higham 2002).
    """
    B = 0.5 * (A + A.T)
    w, V = eigh(B)
    w_clip = np.clip(w, eps, None)
    return (V * w_clip) @ V.T


def _safe_cholesky(R, jitter=1e-12, max_tries=5):
    """
    Robust Cholesky factorization for a possibly nearly-semi-definite
    correlation matrix R. Adds diagonal jitter if needed.
    """
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


def softplus_scalar(x, alpha):
    """
    Scalar Softplus with simple overflow-safe saturation (FD baseline).
    """
    ax = alpha * x
    if ax > 50.0:
        return x
    elif ax < -50.0:
        return 0.0
    else:
        return (math.log(1.0 + math.exp(ax))) / alpha


def safe_exp_ad(x, cutoff=40.0):
    """
    Simple clipping for exp in AAD context to avoid extreme overflow:
        for ADVar, use a linearized exp around +/- cutoff;
        for floats, use math.exp with basic clipping.
    """
    if AAD_AVAILABLE and isinstance(x, ADVar):
        v = x.val
        if v > cutoff:
            # First-order expansion of e^x around x = cutoff.
            return exp(cutoff) * (1.0 + (x - cutoff))
        elif v < -cutoff:
            return exp(-cutoff) * (1.0 + (x + cutoff))
        else:
            return exp(x)
    else:
        v = float(x)
        if v > 700:
            return math.exp(700.0)
        if v < -700:
            return math.exp(-700.0)
        return math.exp(v)


# ============================================================
# Method 1: CE-on-Asset precomputation
# ============================================================
def _precompute_ce_params(sigma, corr, T, j_idx, tiny=1e-16):
    """
    Precompute conditional distribution parameters for CE-on-Asset.

    Let X ~ N(mu, Gamma), where X_i = log S_i(T). We condition on
    X_A = {X_i | i != j}, treat X_j as the 1D conditional variable.

    Then:
        X_j | X_A ~ N(m_cond, s_cond^2)
    with
        m_cond = mu_j + C_vec @ (X_A - mu_A),
        s_cond^2 = Gamma_jj - C_vec @ Gamma_Aj.

    This function returns (C_vec, s_cond, A_indices).
    """
    n = len(sigma)
    A_indices = [i for i in range(n) if i != j_idx]

    # 1. Full covariance matrix Gamma for log-prices:
    #    Gamma_ik = rho_ik * sigma_i * sigma_k * T
    Gamma = (np.outer(sigma, sigma) * corr) * T

    # 2. Partition into A and j blocks
    Gamma_AA = Gamma[np.ix_(A_indices, A_indices)]
    Gamma_Aj = Gamma[A_indices, j_idx]  # shape (n-1,)
    Gamma_jA = Gamma[j_idx, A_indices]  # shape (n-1,)
    Gamma_jj = Gamma[j_idx, j_idx]

    # 3. Solve C_vec @ Gamma_AA = Gamma_jA
    #    (better than explicit inverse)
    try:
        if cho_factor is not None:
            cF = cho_factor(Gamma_AA, lower=True, check_finite=False)
            C_vec = cho_solve(cF, Gamma_jA, check_finite=False)
        else:
            # Solve Gamma_AA^T x^T = Gamma_jA^T, then transpose
            C_vec = np.linalg.solve(Gamma_AA.T, Gamma_jA)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: explicit inverse if Gamma_AA is singular-ish
        try:
            Gamma_AA_inv = np.linalg.inv(Gamma_AA)
            C_vec = Gamma_jA @ Gamma_AA_inv
        except np.linalg.LinAlgError:
            print(f"[Warning] CE precomputation failed: Gamma_AA is singular (j_idx={j_idx}).")
            C_vec = np.zeros(n - 1)

    # 4. Conditional variance
    s2_cond = Gamma_jj - C_vec @ Gamma_Aj
    s_cond = math.sqrt(max(tiny, s2_cond))

    return C_vec, s_cond, A_indices


# ============================================================
# Method 1: CE-on-Asset payoff (analytic in 1 dimension)
# ============================================================
def payoff_ce_ad(
    S_vars, w, K, T, r, sigma, drift, z_std, L, j_idx,
    C_vec, s_cond, A_indices,
    *,
    use_soft_gate=False,
    gate_scale=50.0,
    softplus_alpha=20.0,
    wj_tol=1e-10,
):
    """
    CE-on-Asset payoff implemented with AAD and Softplus fallback.

    Interpretation:
      - Condition on all assets except j (index j_idx).
      - Use the conditional lognormal distribution of S_j(T) given S_A(T)
        to compute the (almost) analytic conditional expectation of
        (w·S_T - K)^+.
      - Optionally blend with a fully softplus-smoothed payoff via a gate.

    Parameters
    ----------
    S_vars : list[ADVar]
        AAD variables for S0_i.
    w : array_like
        Basket weights.
    K : float
        Strike.
    T, r : float
        Maturity and risk-free rate.
    sigma : array_like
        Volatilities.
    drift : array_like
        Log-drift terms (r - 0.5 * sigma^2) * T.
    z_std : array_like
        Standard normal shocks at the factor level.
    L : ndarray
        Cholesky factor of correlation matrix (n x n).
    j_idx : int
        Index of the asset we condition on (the “CE asset”).
    C_vec, s_cond, A_indices :
        Conditional distribution parameters precomputed by _precompute_ce_params.
    use_soft_gate : bool
        If True, blend between CE payoff and softplus payoff.
    gate_scale : float
        Controls how sharply the gate transitions around P(ITM) = 0.5.
    softplus_alpha : float
        Softplus sharpness when using the softplus fallback.
    wj_tol : float
        Tolerance for declaring w_j ≈ 0.

    Returns
    -------
    payoff_ad : ADVar
        AAD-aware payoff at maturity (undiscounted).
    """
    n = len(S_vars)
    sqrtT = math.sqrt(T)
    z_std = np.asarray(z_std, float)

    # 1) Correlated shocks
    z_corr = L @ z_std  # shape (n,)

    wj = float(w[j_idx])

    # 2) If w_j is effectively zero: fall back to the fully-softplus payoff.
    #    NOTE: payoff_softplus_ad is defined in your other module (LRM/Kernel code).
    if abs(wj) < wj_tol:
        return payoff_softplus_ad(  # noqa: F821 (imported from your other file)
            S_vars=S_vars, w=w, K=K, T=T, r=r,
            sigma=sigma, drift=drift, z_std=z_corr,
            alpha=softplus_alpha,
        )

    # 3) Common precomputation: sigma * sqrt(T)
    sigma_sqrtT = sigma * sqrtT

    # 4) Only X_j0 = log S_j0 is treated as ADVar in the conditional mean
    log_Sj0 = log(S_vars[j_idx])          # ADVar
    mu_j_var = log_Sj0 + drift[j_idx]     # E[X_j]

    # 5) Compute B' and the (X_A - mu_A) contribution as a scalar diff_ad
    B_prime_ad = 0.0      # ADVar: depends on S_vars[i] for i in A_indices
    diff_ad = 0.0         # float: depends only on z_corr

    for pos, i in enumerate(A_indices):
        z_i = float(z_corr[i])

        # S_i(T) = S0_i * exp(drift_i + sigma_i * sqrtT * z_i)
        S_i_T = S_vars[i] * exp(drift[i] + sigma_sqrtT[i] * z_i)
        B_prime_ad += w[i] * S_i_T

        # X_i(T) - mu_i = sigma_i * sqrtT * z_i
        diff_ad += C_vec[pos] * (sigma_sqrtT[i] * z_i)

    # 6) Conditional mean and E[S_j(T) | S_A(T)]
    m_cond = mu_j_var + diff_ad          # ADVar
    s2_cond = s_cond * s_cond            # float
    EX_cond = exp(m_cond + 0.5 * s2_cond)  # ADVar

    payoff_ce_exact = None
    prob_ITM = None

    # 7) Analytic CE payoff for a call on S_j with effective strike
    if wj > 0.0:
        # Effective strike on S_j:
        #   Kp = (K - B') / w_j
        Kp = (K - B_prime_ad) / wj
        Kp_scalar = Kp.val if hasattr(Kp, "val") else float(Kp)

        if Kp_scalar <= 0.0:
            # Kp <= 0 → payoff is linear in S_j
            payoff_ce_exact = wj * (EX_cond - Kp)
        else:
            d1 = (m_cond + s2_cond - log(Kp)) / s_cond
            d2 = d1 - s_cond
            payoff_ce_exact = wj * (
                EX_cond * normal_cdf(d1) - Kp * normal_cdf(d2)
            )
            prob_ITM = normal_cdf(d2)  # P(S_j > Kp)
    else:
        # w_j < 0: rewrite as a put on S_j
        u = -wj
        Kminus = (B_prime_ad - K) / u
        Kminus_scalar = Kminus.val if hasattr(Kminus, "val") else float(Kminus)

        if Kminus_scalar <= 0.0:
            # Strike <= 0 → payoff is always zero
            payoff_ce_exact = 0.0 * Kminus
        else:
            d1 = (m_cond + s2_cond - log(Kminus)) / s_cond
            d2 = d1 - s_cond
            payoff_ce_exact = u * (
                Kminus * normal_cdf(-d2) - EX_cond * normal_cdf(-d1)
            )
            prob_ITM = normal_cdf(-d2)  # P(S_j < Kminus)

    # 8) If the gate is disabled, return the analytic CE payoff directly (unbiased).
    if not use_soft_gate:
        return payoff_ce_exact

    # 9) Compute the fully-softplus payoff for the same path
    payoff_soft = payoff_softplus_ad(  # noqa: F821
        S_vars=S_vars, w=w, K=K, T=T, r=r,
        sigma=sigma, drift=drift, z_std=z_corr,
        alpha=softplus_alpha,
    )

    # 10) Smooth gate between CE and softplus based on P(ITM)
    if prob_ITM is None:
        gate = 1.0
    else:
        prob_scalar = prob_ITM.val if hasattr(prob_ITM, "val") else float(prob_ITM)
        centered = prob_scalar - 0.5
        s = 1.0 / (1.0 + math.exp(-centered * gate_scale))
        gate = 2.0 * abs(s - 0.5)

    return gate * payoff_ce_exact + (1.0 - gate) * payoff_soft


def basket_ce_hessian_correlated(
    S0, w, K, T, r, sigma, corr, eps_steps,
    *,
    discount_at_end=True,
    use_soft_gate=False,
    alpha=20.0,
    j_idx=None,
    sanity_print=False,
    tiny=1e-16,
):
    """
    CE-on-Asset + AAD + algo4_optimized:
    Monte Carlo estimator of the Hessian of a correlated basket call
    with respect to S0.

    Timing convention:
      Step 1: Aggregate eps_steps into z_std for each path.
      Step 2: Prepare the AAD computation graph (reset tape, create S_vars).
      Step 3: Forward pass to build CE-on-Asset payoff and discounted price.
      Step 4: Use algo4_optimized to compute the Hessian w.r.t. S0.
      Step 5: Accumulate pathwise contributions into global MC estimates.
    """
    if not AAD_AVAILABLE:
        raise RuntimeError(
            "basket_ce_hessian_correlated requires AAD + algo4_optimized. "
            "Please call it only when AAD_AVAILABLE=True."
        )

    t0_global = time.perf_counter()

    # Basic parameter setup
    S0 = np.asarray(S0, dtype=float)
    w = np.asarray(w, dtype=float)
    sigma_raw = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)
    eps_steps = np.asarray(eps_steps, dtype=float)

    n = S0.shape[0]
    n_paths, n_steps, dim = eps_steps.shape
    if dim != n:
        raise ValueError(
            f"eps_steps.shape[2] = {dim} must equal number of assets n = {n} "
            "so that factors match the correlation matrix."
        )

    sigma_eff = np.maximum(sigma_raw, tiny)
    S0_eff = np.maximum(S0, tiny)

    sqrtT = math.sqrt(T)
    drift = (r - 0.5 * (sigma_eff ** 2.0)) * T
    discount = math.exp(-r * T) if discount_at_end else 1.0
    inv_sqrt_ns = 1.0 / math.sqrt(n_steps)

    # Correlation matrix: Cholesky factor
    L = _safe_cholesky(corr)

    # Precompute CE parameters
    if j_idx is None:
        j_idx = _choose_j_index(w, sigma_eff)

    C_vec, s_cond, A_indices = _precompute_ce_params(
        sigma_eff, corr, T, j_idx, tiny
    )

    price_sum = 0.0
    H_sum = np.zeros((n, n), dtype=float)

    printed_once = False

    # Timing accumulators (seconds)
    t_step1 = t_step2 = t_step3 = t_step4 = t_step5 = 0.0

    for p in range(n_paths):
        t0 = time.perf_counter()

        # Step 1: Aggregate eps_steps in time to form z_std
        z_std = eps_steps[p].sum(axis=0) * inv_sqrt_ns
        t1 = time.perf_counter()

        # Step 2: Prepare AAD tape and variables
        global_tape.reset()
        with use_tape():
            S_vars = [ADVar(float(S0_eff[i])) for i in range(n)]
            t2 = time.perf_counter()

            # Step 3: Build CE-on-Asset payoff and discounted price on the tape
            payoff_ad = payoff_ce_ad(
                S_vars=S_vars,
                w=w,
                K=K,
                T=T,
                r=r,
                sigma=sigma_eff,
                drift=drift,
                z_std=z_std,
                L=L,
                j_idx=j_idx,
                C_vec=C_vec,
                s_cond=s_cond,
                A_indices=A_indices,
                use_soft_gate=use_soft_gate,
                gate_scale=alpha,
                softplus_alpha=alpha,
                wj_tol=1e-10,
            )
            price_ad = discount * payoff_ad
            t3 = time.perf_counter()

        # Step 4: Hessian via algo4_optimized
        H_path = algo4_optimized(price_ad, S_vars)
        t4 = time.perf_counter()

        # Step 5: Accumulate MC estimates
        price_sum += float(price_ad.val)
        H_sum += H_path
        t5 = time.perf_counter()

        t_step1 += (t1 - t0)
        t_step2 += (t2 - t1)
        t_step3 += (t3 - t2)
        t_step4 += (t4 - t3)
        t_step5 += (t5 - t4)

        if sanity_print and (not printed_once):
            print(f"[CE-on-Asset] (S0-space, correlated, j_idx={j_idx}) using AAD + algo4_optimized to compute the Hessian.")
            printed_once = True

    price = price_sum / n_paths
    H = H_sum / n_paths
    elapsed_ms = (time.perf_counter() - t0_global) * 1000.0

    print(f"[Timing] Step 1 For each path, aggregate the stepwise shocks in eps_steps to form the independent standard-normal vector z_std: {t_step1 * 1000.0:.3f} ms")
    print(f"[Timing] Step 2 Prepare the AAD computation graph by resetting the tape and creating the AD variables for S0: {t_step2 * 1000.0:.3f} ms")
    print(f"[Timing] Step 3 In the forward pass, build the CE-on-Asset payoff and the discounted price from S0 and z_std: {t_step3 * 1000.0:.3f} ms")
    print(f"[Timing] Step 4 Use algo4_optimized to compute the Hessian of the price with respect to S0: {t_step4 * 1000.0:.3f} ms")
    print(f"[Timing] Step 5 Accumulate pathwise price and Hessian contributions into the global Monte Carlo estimates: {t_step5 * 1000.0:.3f} ms")

    return price, H, elapsed_ms


# ============================================================
# Method 2: Vibrato variant: CE-on-Asset + 1D Gauss-Hermite quadrature
# ============================================================
from functools import lru_cache


@lru_cache(None)
def _get_herm_gauss_nodes(n_gh):
    """
    Return Gauss-Hermite nodes and weights for n_gh points,
    with weight function exp(-x^2).
    """
    x, w = np.polynomial.hermite.hermgauss(n_gh)
    return x, w


def payoff_ce_vibrato_ad(
    S_vars, w, K, T, r, sigma, drift, z_std, L, j_idx,
    C_vec, s_cond, A_indices,
    *,
    n_inner=8,              # number of Gauss-Hermite points
    softplus_alpha=20.0,
    wj_tol=1e-10,
):
    """
    CE-on-Asset + 1D Gauss–Hermite quadrature (“vibrato-style”) payoff.

    Outer layer:
      - Use the shocks for A_indices to build B' = sum_{i∈A} w_i S_i(T).

    Inner layer:
      - Use the conditional distribution X_j | X_A ~ N(m_cond, s_cond^2)
        and n_inner Gauss–Hermite points to approximate
        E[ softplus( w_j S_j + B' - K ) | X_A ].

    Here we integrate against ξ ~ N(0, 1) via:
      E[f(ξ)] ≈ 1/sqrt(pi) * sum_k w_k f( sqrt(2) x_k ).
    """
    n = len(S_vars)
    sqrtT = math.sqrt(T)

    z_std = np.asarray(z_std, dtype=float)
    z_corr = L @ z_std

    wj = float(w[j_idx])

    # If w_j is effectively zero, fall back to the fully-softplus payoff.
    if abs(wj) < wj_tol:
        return payoff_softplus_ad(  # noqa: F821
            S_vars=S_vars,
            w=w, K=K, T=T, r=r,
            sigma=sigma,
            drift=drift,
            z_std=z_corr,
            alpha=softplus_alpha,
        )

    sigma_sqrtT = sigma * sqrtT

    # Outer: B' and conditional mean m_cond
    B_prime_ad = 0.0  # ADVar
    diff_ad = 0.0     # float

    for pos, i in enumerate(A_indices):
        z_i = float(z_corr[i])

        S_i_T = S_vars[i] * exp(drift[i] + sigma_sqrtT[i] * z_i)
        B_prime_ad += w[i] * S_i_T

        diff_ad += C_vec[pos] * (sigma_sqrtT[i] * z_i)

    log_Sj0 = log(S_vars[j_idx])
    mu_j_var = log_Sj0 + drift[j_idx]
    m_cond = mu_j_var + diff_ad  # ADVar

    # Gauss-Hermite nodes and weights
    gh_x, gh_w = _get_herm_gauss_nodes(int(n_inner))

    payoff_sum_ad = 0.0  # ADVar

    for x_k, w_k in zip(gh_x, gh_w):
        xi = math.sqrt(2.0) * float(x_k)   # ξ_k ~ scaled GH node
        X_j_k = m_cond + s_cond * xi       # ADVar
        S_j_k = safe_exp_ad(X_j_k)

        basket_k = B_prime_ad + wj * S_j_k
        payoff_k = softplus_ad(basket_k - K, softplus_alpha)

        payoff_sum_ad += float(w_k) * payoff_k

    payoff_ad = payoff_sum_ad / math.sqrt(math.pi)

    return payoff_ad


def basket_ce_vibrato_hessian_correlated(
    S0, w, K, T, r, sigma, corr, eps_steps,
    *,
    discount_at_end=True,
    alpha=20.0,
    j_idx=None,
    sanity_print=False,
    tiny=1e-16,
    n_inner=8,
):
    """
    CE-on-Asset + 1D Gauss–Hermite (“vibrato-style”) + AAD + algo4_optimized:
    Monte Carlo estimator of the Hessian with respect to S0.

    Timing convention:
      Step 1: eps_steps -> z_std.
      Step 2: AAD tape + S_vars creation.
      Step 3: Build CE-vibrato payoff and discounted price.
      Step 4: algo4_optimized Hessian.
      Step 5: Monte Carlo accumulation.
    """
    if not AAD_AVAILABLE:
        raise RuntimeError(
            "basket_ce_vibrato_hessian_correlated requires AAD + algo4_optimized. "
            "Please call it only when AAD_AVAILABLE=True."
        )

    t0_global = time.perf_counter()

    S0 = np.asarray(S0, dtype=float)
    w = np.asarray(w, dtype=float)
    sigma_raw = np.asarray(sigma, dtype=float)
    corr = np.asarray(corr, dtype=float)
    eps_steps = np.asarray(eps_steps, dtype=float)

    n = S0.shape[0]
    n_paths, n_steps, dim = eps_steps.shape
    if dim != n:
        raise ValueError(
            f"eps_steps.shape[2] = {dim} must equal number of assets n = {n}."
        )

    sigma_eff = np.maximum(sigma_raw, tiny)
    S0_eff = np.maximum(S0, tiny)

    drift = (r - 0.5 * (sigma_eff ** 2.0)) * T
    discount = math.exp(-r * T) if discount_at_end else 1.0
    inv_sqrt_ns = 1.0 / math.sqrt(n_steps)

    L = _safe_cholesky(corr)

    if j_idx is None:
        j_idx = _choose_j_index(w, sigma_eff)

    C_vec, s_cond, A_indices = _precompute_ce_params(
        sigma_eff, corr, T, j_idx, tiny
    )

    price_sum = 0.0
    H_sum = np.zeros((n, n), dtype=float)
    printed_once = False

    t_step1 = t_step2 = t_step3 = t_step4 = t_step5 = 0.0

    for p in range(n_paths):
        t0 = time.perf_counter()

        # Step 1: eps_steps -> z_std
        z_std = eps_steps[p].sum(axis=0) * inv_sqrt_ns
        t1 = time.perf_counter()

        # Step 2: prepare AAD tape
        global_tape.reset()
        with use_tape():
            S_vars = [ADVar(float(S0_eff[i])) for i in range(n)]
            t2 = time.perf_counter()

            # Step 3: CE-vibrato payoff via Gauss–Hermite
            payoff_ad = payoff_ce_vibrato_ad(
                S_vars=S_vars,
                w=w,
                K=K,
                T=T,
                r=r,
                sigma=sigma_eff,
                drift=drift,
                z_std=z_std,
                L=L,
                j_idx=j_idx,
                C_vec=C_vec,
                s_cond=s_cond,
                A_indices=A_indices,
                n_inner=n_inner,
                softplus_alpha=alpha,
                wj_tol=1e-10,
            )
            price_ad = discount * payoff_ad
            t3 = time.perf_counter()

        # Step 4: Hessian via algo4_optimized
        H_path = algo4_optimized(price_ad, S_vars)
        t4 = time.perf_counter()

        # Step 5: MC accumulation
        price_sum += float(price_ad.val)
        H_sum += H_path
        t5 = time.perf_counter()

        t_step1 += (t1 - t0)
        t_step2 += (t2 - t1)
        t_step3 += (t3 - t2)
        t_step4 += (t4 - t3)
        t_step5 += (t5 - t4)

        if sanity_print and (not printed_once):
            print(f"[CE-vibrato] (S0-space, correlated, j_idx={j_idx}) using CE-on-Asset + Gauss–Hermite + AAD + algo4_optimized to compute the Hessian.")
            printed_once = True

    price = price_sum / n_paths
    H = H_sum / n_paths
    elapsed_ms = (time.perf_counter() - t0_global) * 1000.0

    print(f"[Timing] Step 1 For each path, aggregate the time-step shocks in eps_steps to form the independent standard-normal vector z_std: {t_step1 * 1000.0:.3f} ms")
    print(f"[Timing] Step 2 Prepare the AAD computation graph by resetting the tape and creating the AD variables for S0: {t_step2 * 1000.0:.3f} ms")
    print(f"[Timing] Step 3 In the forward pass, build the CE-with-Gauss–Hermite payoff (vibrato-style) and the discounted price from S0 and z_std: {t_step3 * 1000.0:.3f} ms")
    print(f"[Timing] Step 4 Use algo4_optimized to compute the Hessian of the price with respect to S0: {t_step4 * 1000.0:.3f} ms")
    print(f"[Timing] Step 5 Accumulate pathwise price and Hessian contributions into the global Monte Carlo estimates: {t_step5 * 1000.0:.3f} ms")

    return price, H, elapsed_ms


# ============================================================
# Method 3: Unified FD benchmark
# ============================================================
def basket_fd_hessian(
    S0, w, K, T, r, sigma, L, eps_steps,
    alpha=25,
    h_rel=1e-4,
    tiny=1e-16,
):
    """
    Finite-difference Hessian with respect to S0.

    Uses:
      - Monte Carlo pricing with Softplus-smoothed basket payoff
      - Central differences along each dimension and cross terms

    Timing convention:
      Step 1: For each path, generate z_std and z_corr.
      Step 2: Build terminal asset vector ST_vec.
      Step 3: Compute basket value and Softplus payoff.
      Step 4: Discount payoff and accumulate MC price.
      Step 5: Assemble the full FD Hessian via shifted evaluations.
    """
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

    # Timing accumulators
    t_step1 = t_step2 = t_step3 = t_step4 = t_step5 = 0.0

    def price_fn(S_vec):
        """
        Monte Carlo price for a given S0 vector,
        with Softplus-smoothed basket call payoff.
        """
        nonlocal t_step1, t_step2, t_step3, t_step4

        total = 0.0
        drift_log = (r - 0.5 * (sigma_eff ** 2.0)) * T
        S_vec_eff = np.maximum(S_vec, tiny)

        for p in range(n_paths):
            t0_p = time.perf_counter()

            # Step 1: generate z_std, z_corr
            z_std = eps_steps[p].sum(axis=0) * inv_sqrt_ns
            z_corr = L @ z_std
            t1_p = time.perf_counter()

            # Step 2: build terminal asset vector ST_vec
            ST_vec = S_vec_eff * np.exp(drift_log + sigma_eff * sqrtT * z_corr)
            t2_p = time.perf_counter()

            # Step 3: compute basket and Softplus payoff
            basket = float(np.dot(w, ST_vec))
            payoff = softplus_scalar(basket - K, alpha=alpha)
            t3_p = time.perf_counter()

            # Step 4: discount and accumulate
            total += discount * payoff
            t4_p = time.perf_counter()

            t_step1 += (t1_p - t0_p)
            t_step2 += (t2_p - t1_p)
            t_step3 += (t3_p - t2_p)
            t_step4 += (t4_p - t3_p)

        return total / n_paths

    # Central price
    price = price_fn(S0)
    H = np.zeros((n, n))

    # Step 5: FD Hessian assembly (diagonal + off-diagonal)
    t5_start = time.perf_counter()

    # Diagonal terms
    for i in range(n):
        S_up = S0.copy(); S_up[i] += h[i]
        S_dn = S0.copy(); S_dn[i] -= h[i]
        f_up = price_fn(S_up)
        f_dn = price_fn(S_dn)
        H[i, i] = (f_up - 2 * price + f_dn) / (h[i] ** 2)

    # Cross terms
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

    print(f"[Timing] Step 1 For each Monte Carlo path, generate the independent normals and apply the Cholesky factor L to obtain the correlated shocks: {t_step1 * 1000.0:.3f} ms")
    print(f"[Timing] Step 2 For each path, generate the terminal asset vector ST_vec from the current S0 input and the correlated shocks: {t_step2 * 1000.0:.3f} ms")
    print(f"[Timing] Step 3 For each path, compute the basket value and the smoothed Softplus payoff: {t_step3 * 1000.0:.3f} ms")
    print(f"[Timing] Step 4 Discount the payoff and accumulate it into the Monte Carlo price estimate: {t_step4 * 1000.0:.3f} ms")
    print(f"[Timing] Step 5 Assemble the full finite-difference Hessian with respect to S0 by repeatedly evaluating the pricing function at shifted points: {t_step5 * 1000.0:.3f} ms")

    return price, H, elapsed_ms, n_evals


# ============================================================
# AAD stubs (for environments without aad_edge_pushing)
# ============================================================
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
# Example program: CE vs Vibrato vs FD
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print(" CE (Conditional Expectation + AAD) vs FD (Finite Difference) Hessian comparison")
    print("=" * 60)

    # 1. Market and option parameters
    n_assets = 8
    n_assets_global = n_assets  # for AAD stubs

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

    # 2. Monte Carlo parameters
    n_paths = 1000
    n_steps = 1

    # 3. Random numbers
    np.random.seed(42)
    eps_steps = np.random.randn(n_paths, n_steps, n_assets)

    # 4. Cholesky factor for FD baseline
    L = _safe_cholesky(corr)

    # 5. CE-on-Asset estimator
    print("\n" + "-" * 20 + " 1. CE (Conditional Expectation + AAD) " + "-" * 20)
    try:
        price_ce, H_ce, elapsed_ce = basket_ce_hessian_correlated(
            S0, w, K, T, r, sigma, corr, eps_steps,
            discount_at_end=True,
            use_soft_gate=False,
            alpha=200,
            sanity_print=True,
        )
        print(f"N_paths: {n_paths}")
        print(f"Total time: {elapsed_ce:.2f} ms")
        print(f"Estimated price (CE): {price_ce:.6f}")
        print("Estimated Hessian (H_CE):")
        with np.printoptions(precision=6, suppress=True):
            print(H_ce)

    except Exception as e:
        print(f"\nError when running CE: {e}")
        import traceback
        traceback.print_exc()
        price_ce, H_ce, elapsed_ce = 0.0, np.zeros((n_assets, n_assets)), -1.0

    # 6. CE-vibrato estimator
    print("\n" + "-" * 20 + " 2. CE-vibrato (Gauss–Hermite + AAD) " + "-" * 20)
    try:
        price_vib, H_vib, elapsed_vib = basket_ce_vibrato_hessian_correlated(
            S0, w, K, T, r, sigma, corr, eps_steps,
            discount_at_end=True,
            alpha=200,
            sanity_print=True,
        )
        print(f"N_paths: {n_paths}")
        print(f"Total time: {elapsed_vib:.2f} ms")
        print(f"Estimated price (CE-vibrato): {price_vib:.6f}")
        print("Estimated Hessian (H_CE_vibrato):")
        with np.printoptions(precision=6, suppress=True):
            print(H_vib)

    except Exception as e:
        print(f"\nError when running CE-vibrato: {e}")
        import traceback
        traceback.print_exc()
        price_vib, H_vib, elapsed_vib = 0.0, np.zeros((n_assets, n_assets)), -1.0

    # 7. FD baseline
    print("\n" + "-" * 20 + " 3. FD (benchmark) " + "-" * 20)

    fd_n_evals = 1 + 2 * n_assets + 4 * (n_assets * (n_assets - 1) // 2)
    print(f"[Note] FD is comparatively slow; it requires {fd_n_evals} pricing evaluations...")

    fd_alpha = 25
    fd_h_rel = 1e-4

    try:
        price_fd, H_fd, elapsed_fd, n_evals = basket_fd_hessian(
            S0, w, K, T, r, sigma, L, eps_steps,
            alpha=fd_alpha,
            h_rel=fd_h_rel,
        )

        print(f"N_paths: {n_paths}")
        print(f"FD calls: {n_evals}")
        print(f"FD Softplus alpha: {fd_alpha}")
        print(f"Total time: {elapsed_fd:.2f} ms")
        print(f"Estimated price (FD-Softplus): {price_fd:.6f}")
        print("Estimated Hessian (H_FD):")
        with np.printoptions(precision=6, suppress=True):
            print(H_fd)

    except Exception as e:
        print(f"\nError when running FD (benchmark): {e}")
        import traceback
        traceback.print_exc()
        price_fd, H_fd, elapsed_fd = 0.0, np.zeros((n_assets, n_assets)), -1.0

    # 8. Final comparison
    print("\n" + "=" * 60)
    print("--- Final comparison (Hessian differences vs FD) ---")
    print("=" * 60)

    if elapsed_fd <= 0:
        print("FD benchmark failed; cannot compare CE methods to FD.")
    else:
        if elapsed_ce > 0:
            diff_ce = H_ce - H_fd
            norm_ce = norm(diff_ce)
            print(f"||H_CE - H_FD||_F (Frobenius norm): {norm_ce:.6e}")
            print(f"  (CE time: {elapsed_ce:.2f} ms vs FD: {elapsed_fd:.2f} ms, "
                  f"CE is about {elapsed_fd / elapsed_ce:.1f}x faster)")

        if elapsed_vib > 0:
            diff_vib = H_vib - H_fd
            norm_vib = norm(diff_vib)
            print(f"||H_CE_vibrato - H_FD||_F (Frobenius norm): {norm_vib:.6e}")
            print(f"  (CE-vibrato time: {elapsed_vib:.2f} ms vs FD: {elapsed_fd:.2f} ms, "
                  f"CE-vibrato is about {elapsed_fd / elapsed_vib:.1f}x faster)")
