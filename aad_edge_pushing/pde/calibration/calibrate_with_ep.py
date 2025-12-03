"""
Volatility Surface Calibration with Edge-Pushing Hessian

Compares EP Hessian vs Numerical Hessian for accuracy and performance.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from pathlib import Path

# Add paths - now in pde/calibration/, need to go up to AAD root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import EP engine
from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape, use_tape
from aad_edge_pushing.aad.core.engine import edge_push_hessian

# ============================================================================
# Black-Scholes Functions
# ============================================================================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r, option_type='C', bounds=(0.01, 3.0)):
    """Compute implied volatility via root finding."""
    if T <= 0:
        return np.nan

    price_func = bs_call_price if option_type == 'C' else bs_put_price
    intrinsic = max(S - K, 0) if option_type == 'C' else max(K - S, 0)

    if market_price <= intrinsic:
        return np.nan

    def objective(sigma):
        return price_func(S, K, T, r, sigma) - market_price

    try:
        low_val = objective(bounds[0])
        high_val = objective(bounds[1])
        if low_val * high_val > 0:
            return np.nan
        return brentq(objective, bounds[0], bounds[1])
    except:
        return np.nan


# ============================================================================
# Data Loading
# ============================================================================

def load_options_data(filepath, quote_date='2025-02-06'):
    """Load and preprocess options data."""
    df = pd.read_csv(filepath)
    S0 = df['underlying_bid_eod'].iloc[0]

    df = df[
        (df['open_interest'] >= 100) &
        (df['bid_eod'] > 0) &
        (df['ask_eod'] > df['bid_eod'])
    ].copy()

    df['mid_price'] = (df['bid_eod'] + df['ask_eod']) / 2
    df['expiration'] = pd.to_datetime(df['expiration'])
    quote_dt = pd.to_datetime(quote_date)
    df['T'] = (df['expiration'] - quote_dt).dt.days / 365.0
    df = df[df['T'] > 0.01]

    return df, S0


def compute_implied_vols(df, S0, r=0.045):
    """Compute implied volatilities for all options."""
    ivs = []
    for idx, row in df.iterrows():
        iv = implied_vol(row['mid_price'], S0, row['strike'], row['T'], r, row['option_type'])
        ivs.append(iv)

    df['iv'] = ivs
    df = df[~np.isnan(df['iv'])]
    df = df[(df['iv'] > 0.01) & (df['iv'] < 2.0)]
    df['moneyness'] = df['strike'] / S0

    return df


# ============================================================================
# B-Spline Volatility Surface
# ============================================================================

def create_bspline_vol_surface(moneyness_grid, T_grid, degree=3):
    """Create B-spline specification for volatility surface."""
    n_m = len(moneyness_grid) + degree - 1
    n_T = len(T_grid) + degree - 1

    knots_m = np.concatenate([
        [moneyness_grid[0]] * degree,
        moneyness_grid,
        [moneyness_grid[-1]] * degree
    ])

    knots_T = np.concatenate([
        [T_grid[0]] * degree,
        T_grid,
        [T_grid[-1]] * degree
    ])

    return {
        'knots_m': knots_m,
        'knots_T': knots_T,
        'degree': degree,
        'n_m': n_m,
        'n_T': n_T,
        'm_range': (moneyness_grid[0], moneyness_grid[-1]),
        'T_range': (T_grid[0], T_grid[-1])
    }


def eval_bspline_surface(m, T, coeffs, bspline_spec):
    """Evaluate B-spline surface at (m, T)."""
    knots_m = bspline_spec['knots_m']
    knots_T = bspline_spec['knots_T']
    degree = bspline_spec['degree']
    n_m = bspline_spec['n_m']
    n_T = bspline_spec['n_T']

    result = 0.0
    for k in range(n_m):
        c_m = np.zeros(n_m)
        c_m[k] = 1.0
        B_m_k = BSpline(knots_m, c_m, degree)(m)

        for j in range(n_T):
            c_T = np.zeros(n_T)
            c_T[j] = 1.0
            B_T_j = BSpline(knots_T, c_T, degree)(T)

            result += coeffs[k, j] * B_m_k * B_T_j

    return result


# ============================================================================
# Calibration with EP Hessian
# ============================================================================

def calibrate_vol_surface_ep(df, S0, bspline_spec, use_ep=True):
    """Calibrate volatility surface with EP Hessian."""
    n_m, n_T = bspline_spec['n_m'], bspline_spec['n_T']
    n_params = n_m * n_T

    mean_iv = df['iv'].mean()
    x0 = np.full(n_params, mean_iv)

    m_data = df['moneyness'].values
    T_data = df['T'].values
    iv_data = df['iv'].values
    n_data = len(df)

    print(f"\nCalibration setup:")
    print(f"  Parameters: {n_params} ({n_m} x {n_T})")
    print(f"  Data points: {n_data}")
    print(f"  Mean IV: {mean_iv:.4f}")

    # Precompute basis functions
    knots_m = bspline_spec['knots_m']
    knots_T = bspline_spec['knots_T']
    degree = bspline_spec['degree']

    B_m_mat = np.zeros((n_data, n_m))
    B_T_mat = np.zeros((n_data, n_T))

    for k in range(n_m):
        c = np.zeros(n_m)
        c[k] = 1.0
        B_m_mat[:, k] = BSpline(knots_m, c, degree)(m_data)

    for j in range(n_T):
        c = np.zeros(n_T)
        c[j] = 1.0
        B_T_mat[:, j] = BSpline(knots_T, c, degree)(T_data)

    # Precompute Φ[i, p] = B_m[i, k] * B_T[i, j] where p = k * n_T + j
    Phi = np.zeros((n_data, n_params))
    for k in range(n_m):
        for j in range(n_T):
            p = k * n_T + j
            Phi[:, p] = B_m_mat[:, k] * B_T_mat[:, j]

    def objective(x):
        """Sum of squared errors: 0.5 * ||Φw - iv||²"""
        sigma_model = Phi @ x
        errors = sigma_model - iv_data
        return 0.5 * np.sum(errors ** 2)

    def gradient(x):
        """Gradient: Φᵀ(Φw - iv)"""
        sigma_model = Phi @ x
        errors = sigma_model - iv_data
        return Phi.T @ errors

    # Calibrate using L-BFGS-B
    print("\nRunning L-BFGS-B optimization...")
    t0 = time.time()

    result = minimize(
        objective, x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=[(0.01, 1.5)] * n_params,
        options={'maxiter': 500, 'disp': True}
    )

    t_opt = time.time() - t0
    print(f"\nOptimization time: {t_opt:.2f}s")
    print(f"Final objective: {result.fun:.6f}")
    print(f"RMSE: {np.sqrt(2*result.fun/n_data):.4f}")

    # Compute Hessians
    x_opt = result.x

    # ===== Numerical Hessian =====
    print("\nComputing numerical Hessian...")
    t0 = time.time()
    eps = 1e-5
    H_num = np.zeros((n_params, n_params))
    g0 = gradient(x_opt)
    for i in range(n_params):
        x_plus = x_opt.copy()
        x_plus[i] += eps
        g_plus = gradient(x_plus)
        H_num[i, :] = (g_plus - g0) / eps
    H_num = 0.5 * (H_num + H_num.T)
    t_num = time.time() - t0
    print(f"  Time: {t_num:.4f}s")

    # ===== EP Hessian =====
    H_ep = None
    t_ep = None

    if use_ep:
        print("\nComputing EP Hessian...")
        t0 = time.time()

        def objective_ad(inputs):
            """AD-compatible objective function using dict inputs."""
            # inputs is a dict: {'w0': ADVar, 'w1': ADVar, ...}
            total = inputs['w0'] * 0.0  # Initialize as ADVar with value 0

            for i in range(n_data):
                # σ_model[i] = Σₚ Φ[i,p] * wₚ
                sigma_i = inputs['w0'] * 0.0  # Initialize
                for p in range(n_params):
                    sigma_i = sigma_i + Phi[i, p] * inputs[f'w{p}']

                error_i = sigma_i - iv_data[i]
                total = total + error_i * error_i

            return total * 0.5

        # Convert to dict format
        inputs_dict = {f'w{p}': x_opt[p] for p in range(n_params)}

        try:
            H_ep = edge_push_hessian(objective_ad, inputs_dict, sparse=False)
            t_ep = time.time() - t0
            print(f"  Time: {t_ep:.4f}s")
        except Exception as e:
            print(f"  EP failed: {e}")
            t_ep = None

    # Compare results
    print("\n" + "="*60)
    print("HESSIAN COMPARISON")
    print("="*60)

    print(f"\nNumerical Hessian:")
    print(f"  Shape: {H_num.shape}")
    print(f"  Time: {t_num:.4f}s")
    print(f"  Condition: {np.linalg.cond(H_num):.2e}")

    if H_ep is not None:
        print(f"\nEP Hessian:")
        print(f"  Shape: {H_ep.shape}")
        print(f"  Time: {t_ep:.4f}s")
        print(f"  Condition: {np.linalg.cond(H_ep):.2e}")

        # Compare
        diff = np.abs(H_ep - H_num)
        rel_err = diff / (np.abs(H_num) + 1e-10)

        print(f"\nDifference (EP - Numerical):")
        print(f"  Max abs error: {np.max(diff):.2e}")
        print(f"  Mean abs error: {np.mean(diff):.2e}")
        print(f"  Max rel error: {np.max(rel_err):.2e}")
        print(f"  Mean rel error: {np.mean(rel_err):.2e}")

        if t_ep is not None:
            print(f"\nSpeedup: {t_num/t_ep:.2f}x" if t_ep > 0 else "")

    return {
        'coeffs': result.x.reshape(n_m, n_T),
        'result': result,
        'H_num': H_num,
        'H_ep': H_ep,
        't_num': t_num,
        't_ep': t_ep,
        'bspline_spec': bspline_spec,
        't_opt': t_opt,
        'iv_data': iv_data,
        'Phi': Phi
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_hessian_comparison(calib_result, save_path=None):
    """Plot Hessian comparison: Numerical vs EP."""
    H_num = calib_result['H_num']
    H_ep = calib_result.get('H_ep')

    ncols = 3 if H_ep is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 4))

    # Numerical Hessian
    ax = axes[0]
    im = ax.imshow(np.log10(np.abs(H_num) + 1e-10), cmap='hot', aspect='auto')
    ax.set_title('Numerical Hessian (log₁₀|H|)')
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Parameter Index')
    fig.colorbar(im, ax=ax, shrink=0.8)

    if H_ep is not None:
        # EP Hessian
        ax = axes[1]
        im = ax.imshow(np.log10(np.abs(H_ep) + 1e-10), cmap='hot', aspect='auto')
        ax.set_title('EP Hessian (log₁₀|H|)')
        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('Parameter Index')
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Difference
        ax = axes[2]
        diff = np.abs(H_ep - H_num)
        im = ax.imshow(np.log10(diff + 1e-15), cmap='RdBu_r', aspect='auto')
        ax.set_title('|EP - Numerical| (log₁₀)')
        ax.set_xlabel('Parameter Index')
        ax.set_ylabel('Parameter Index')
        fig.colorbar(im, ax=ax, shrink=0.8)
    else:
        # Just show condition number info
        ax = axes[1]
        eigvals = np.linalg.eigvalsh(H_num)
        ax.bar(range(len(eigvals)), np.sort(eigvals)[::-1])
        ax.set_title('Hessian Eigenvalues')
        ax.set_xlabel('Index')
        ax.set_ylabel('Eigenvalue')
        ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.close()


def plot_vol_surface(calib_result, df, S0, save_path=None):
    """Plot the calibrated volatility surface."""
    coeffs = calib_result['coeffs']
    bspline_spec = calib_result['bspline_spec']

    m_range = np.linspace(0.85, 1.15, 50)
    T_range = np.linspace(0.02, 0.8, 50)
    M, T = np.meshgrid(m_range, T_range)

    Z = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Z[i, j] = eval_bspline_surface(M[i, j], T[i, j], coeffs, bspline_spec)

    fig = plt.figure(figsize=(14, 5))

    # 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(M, T, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Moneyness (K/S)')
    ax1.set_ylabel('Time to Expiry (T)')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('B-Spline Volatility Surface')
    ax1.view_init(elev=25, azim=-60)
    fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.1)

    # Contour
    ax2 = fig.add_subplot(132)
    cs = ax2.contourf(M, T, Z, levels=20, cmap='viridis')
    ax2.scatter(df['moneyness'], df['T'], c=df['iv'], s=10, cmap='viridis',
                edgecolors='white', linewidth=0.3)
    ax2.set_xlabel('Moneyness (K/S)')
    ax2.set_ylabel('Time to Expiry (T)')
    ax2.set_title('Volatility Contours + Market Data')
    fig.colorbar(cs, ax=ax2)

    # Numerical Hessian
    ax3 = fig.add_subplot(133)
    H = calib_result['H_num']
    im = ax3.imshow(np.log10(np.abs(H) + 1e-10), cmap='hot', aspect='auto')
    ax3.set_xlabel('Parameter Index')
    ax3.set_ylabel('Parameter Index')
    ax3.set_title('Hessian log₁₀|H| Heatmap')
    fig.colorbar(im, ax=ax3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")

    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("VOLATILITY SURFACE CALIBRATION WITH EP HESSIAN")
    print("=" * 70)

    # Load data
    print("\n1. Loading options data...")
    df, S0 = load_options_data('UnderlyingOptionsEODQuotes_2025-02-06.csv')
    print(f"   Underlying: {S0:.2f}")
    print(f"   Options: {len(df):,}")

    # Compute implied vols
    print("\n2. Computing implied volatilities...")
    df = compute_implied_vols(df, S0)
    print(f"   Valid IVs: {len(df):,}")

    # Filter
    df = df[(df['moneyness'] > 0.8) & (df['moneyness'] < 1.2)]
    df = df[df['T'] < 1.0]
    print(f"   Filtered: {len(df):,}")

    # Use smaller grid for EP testing (fewer params = faster EP)
    print("\n3. Setting up B-spline surface (small grid for EP test)...")
    moneyness_grid = np.array([0.9, 1.0, 1.1])  # 3 points
    T_grid = np.array([0.05, 0.25])             # 2 points
    bspline_spec = create_bspline_vol_surface(moneyness_grid, T_grid, degree=3)
    n_params = bspline_spec['n_m'] * bspline_spec['n_T']
    print(f"   Grid: {len(moneyness_grid)} x {len(T_grid)}")
    print(f"   Parameters: {n_params}")

    # Limit data points for faster EP
    max_data = 100
    if len(df) > max_data:
        df = df.sample(n=max_data, random_state=42)
        print(f"   Sampled to {len(df)} data points for EP test")

    # Calibrate with EP
    print("\n4. Calibrating with EP Hessian...")
    calib_result = calibrate_vol_surface_ep(df, S0, bspline_spec, use_ep=True)

    # Visualize
    print("\n5. Generating visualizations...")
    plot_vol_surface(calib_result, df, S0, save_path='vol_surface_ep.png')
    plot_hessian_comparison(calib_result, save_path='hessian_comparison.png')

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return calib_result, df, S0


if __name__ == "__main__":
    result, df, S0 = main()
