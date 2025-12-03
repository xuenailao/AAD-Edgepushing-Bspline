"""
Improved Volatility Surface Calibration

Key improvements for short-term options:
1. Non-uniform T grid with more knots at short maturities
2. Weighted least squares (higher weight for short-term)
3. Comparison of uniform vs improved grids
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path

# Add paths - now in pde/calibration/, need to go up to AAD root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# ============================================================================
# Black-Scholes Functions
# ============================================================================

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r, option_type='C', bounds=(0.01, 3.0)):
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
# Calibration with Weighted Least Squares
# ============================================================================

def calibrate_vol_surface(df, S0, bspline_spec, weights=None):
    """Calibrate with optional weights."""
    n_m, n_T = bspline_spec['n_m'], bspline_spec['n_T']
    n_params = n_m * n_T

    mean_iv = df['iv'].mean()
    x0 = np.full(n_params, mean_iv)

    m_data = df['moneyness'].values
    T_data = df['T'].values
    iv_data = df['iv'].values
    n_data = len(df)

    if weights is None:
        weights = np.ones(n_data)
    weights = weights / weights.sum() * n_data  # Normalize

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

    # Φ[i, p] = B_m[i, k] * B_T[i, j]
    Phi = np.zeros((n_data, n_params))
    for k in range(n_m):
        for j in range(n_T):
            p = k * n_T + j
            Phi[:, p] = B_m_mat[:, k] * B_T_mat[:, j]

    def objective(x):
        sigma_model = Phi @ x
        errors = sigma_model - iv_data
        return 0.5 * np.sum(weights * errors ** 2)

    def gradient(x):
        sigma_model = Phi @ x
        errors = sigma_model - iv_data
        return Phi.T @ (weights * errors)

    t0 = time.time()
    result = minimize(
        objective, x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=[(0.01, 1.5)] * n_params,
        options={'maxiter': 500, 'disp': False}
    )
    t_opt = time.time() - t0

    # Compute RMSE by T bucket
    sigma_model = Phi @ result.x
    errors = sigma_model - iv_data

    T_buckets = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]
    rmse_by_T = {}
    for T_lo, T_hi in T_buckets:
        mask = (T_data >= T_lo) & (T_data < T_hi)
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean(errors[mask]**2))
            rmse_by_T[f'{T_lo:.2f}-{T_hi:.2f}'] = rmse

    return {
        'coeffs': result.x.reshape(n_m, n_T),
        'result': result,
        'bspline_spec': bspline_spec,
        't_opt': t_opt,
        'rmse_total': np.sqrt(np.mean(errors**2)),
        'rmse_by_T': rmse_by_T,
        'Phi': Phi,
        'iv_data': iv_data,
        'T_data': T_data,
        'm_data': m_data
    }


# ============================================================================
# Main Comparison
# ============================================================================

def main():
    print("=" * 70)
    print("IMPROVED VOLATILITY SURFACE CALIBRATION")
    print("=" * 70)

    # Load data
    print("\n1. Loading options data...")
    df, S0 = load_options_data('UnderlyingOptionsEODQuotes_2025-02-06.csv')
    print(f"   Underlying: {S0:.2f}")

    # Compute implied vols
    print("\n2. Computing implied volatilities...")
    df = compute_implied_vols(df, S0)

    # Filter
    df = df[(df['moneyness'] > 0.8) & (df['moneyness'] < 1.2)]
    df = df[df['T'] < 1.0]
    print(f"   Valid options: {len(df):,}")

    # =========================================================================
    # Grid Configurations to Compare
    # =========================================================================

    configs = {
        'Uniform (5x4)': {
            'm_grid': np.array([0.85, 0.95, 1.0, 1.05, 1.15]),
            'T_grid': np.array([0.02, 0.1, 0.25, 0.5]),
            'weights': None
        },
        'Dense Short-T (5x6)': {
            'm_grid': np.array([0.85, 0.95, 1.0, 1.05, 1.15]),
            'T_grid': np.array([0.02, 0.04, 0.08, 0.15, 0.3, 0.6]),  # More knots at short T
            'weights': None
        },
        'Dense M + Short-T (7x6)': {
            'm_grid': np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]),
            'T_grid': np.array([0.02, 0.04, 0.08, 0.15, 0.3, 0.6]),
            'weights': None
        },
        'Weighted Short-T (5x4)': {
            'm_grid': np.array([0.85, 0.95, 1.0, 1.05, 1.15]),
            'T_grid': np.array([0.02, 0.1, 0.25, 0.5]),
            'weights': 'short_T'  # Will compute weight = 1/sqrt(T)
        },
    }

    results = {}

    print("\n3. Calibrating different configurations...")
    print("-" * 70)

    for name, cfg in configs.items():
        print(f"\n   {name}:")

        bspline_spec = create_bspline_vol_surface(cfg['m_grid'], cfg['T_grid'], degree=3)
        n_params = bspline_spec['n_m'] * bspline_spec['n_T']
        print(f"      Parameters: {n_params}")

        # Compute weights if needed
        if cfg['weights'] == 'short_T':
            weights = 1.0 / np.sqrt(df['T'].values + 0.01)  # More weight for short T
        else:
            weights = None

        calib = calibrate_vol_surface(df, S0, bspline_spec, weights)

        print(f"      Total RMSE: {calib['rmse_total']:.4f}")
        print(f"      RMSE by T:")
        for bucket, rmse in calib['rmse_by_T'].items():
            print(f"         T∈[{bucket}]: {rmse:.4f}")
        print(f"      Time: {calib['t_opt']:.2f}s")

        results[name] = calib

    # =========================================================================
    # Visualization
    # =========================================================================

    print("\n4. Generating comparison plots...")

    # Plot volatility smiles comparison at T = 0.05
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    T_slices = [0.05, 0.10, 0.25, 0.50]
    m_range = np.linspace(0.85, 1.15, 100)

    for idx, T_target in enumerate(T_slices):
        ax = axes[idx // 2, idx % 2]

        # Market data
        T_tol = 0.03
        mask = (df['T'] > T_target - T_tol) & (df['T'] < T_target + T_tol)
        df_slice = df[mask]
        if len(df_slice) > 0:
            ax.scatter(df_slice['moneyness'], df_slice['iv'], c='red', s=30,
                      label='Market', zorder=5, alpha=0.6)

        # Model curves
        colors = ['blue', 'green', 'purple', 'orange']
        for (name, calib), color in zip(results.items(), colors):
            coeffs = calib['coeffs']
            bspline_spec = calib['bspline_spec']
            sigma_model = [eval_bspline_surface(m, T_target, coeffs, bspline_spec) for m in m_range]
            ax.plot(m_range, sigma_model, color=color, linewidth=2, label=name, alpha=0.8)

        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'Volatility Smile at T = {T_target:.2f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.85, 1.15)

    plt.tight_layout()
    plt.savefig('vol_smiles_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: vol_smiles_comparison.png")
    plt.close()

    # Plot RMSE comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(results))
    width = 0.15

    T_buckets = ['0.00-0.10', '0.10-0.25', '0.25-0.50', '0.50-1.00']
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']

    for i, bucket in enumerate(T_buckets):
        rmses = [results[name]['rmse_by_T'].get(bucket, 0) for name in results]
        ax.bar(x + i * width, rmses, width, label=f'T∈[{bucket}]', color=colors[i])

    ax.set_xlabel('Configuration')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE Comparison by Time-to-Expiry Bucket')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results.keys(), rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('rmse_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: rmse_comparison.png")
    plt.close()

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<25} {'Params':>8} {'RMSE':>10} {'Short-T RMSE':>12}")
    print("-" * 60)
    for name, calib in results.items():
        n_params = calib['bspline_spec']['n_m'] * calib['bspline_spec']['n_T']
        short_rmse = calib['rmse_by_T'].get('0.00-0.10', np.nan)
        print(f"{name:<25} {n_params:>8} {calib['rmse_total']:>10.4f} {short_rmse:>12.4f}")

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

    return results, df, S0


if __name__ == "__main__":
    results, df, S0 = main()
