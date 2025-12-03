"""
Fast Volatility Surface Calibration - Without Edge-Pushing Hessian

This script:
1. Loads real SPX options data
2. Computes implied volatilities using Black-Scholes
3. Fits a B-spline volatility surface σ(K/S, T)
4. Visualizes the volatility surface
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import time
import sys
from pathlib import Path

# Add paths - now in pde/calibration/, need to go up to AAD root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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
        # Check if solution exists in bounds
        low_val = objective(bounds[0])
        high_val = objective(bounds[1])

        if low_val * high_val > 0:
            return np.nan

        return brentq(objective, bounds[0], bounds[1])
    except:
        return np.nan


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_options_data(filepath, quote_date='2025-02-06'):
    """Load and preprocess options data."""
    df = pd.read_csv(filepath)

    # Get underlying price
    S0 = df['underlying_bid_eod'].iloc[0]

    # Filter for valid options
    df = df[
        (df['open_interest'] >= 100) &
        (df['bid_eod'] > 0) &
        (df['ask_eod'] > df['bid_eod'])
    ].copy()

    # Compute mid price
    df['mid_price'] = (df['bid_eod'] + df['ask_eod']) / 2

    # Convert expiration to datetime and compute time to expiry
    df['expiration'] = pd.to_datetime(df['expiration'])
    quote_dt = pd.to_datetime(quote_date)
    df['T'] = (df['expiration'] - quote_dt).dt.days / 365.0

    # Filter out expired options
    df = df[df['T'] > 0.01]  # At least 4 days to expiry

    return df, S0


def compute_implied_vols(df, S0, r=0.045):
    """Compute implied volatilities for all options."""
    ivs = []

    for idx, row in df.iterrows():
        K = row['strike']
        T = row['T']
        price = row['mid_price']
        opt_type = row['option_type']

        iv = implied_vol(price, S0, K, T, r, opt_type)
        ivs.append(iv)

    df['iv'] = ivs

    # Filter valid IVs
    df = df[~np.isnan(df['iv'])]
    df = df[(df['iv'] > 0.01) & (df['iv'] < 2.0)]

    # Compute moneyness
    df['moneyness'] = df['strike'] / S0

    return df


# ============================================================================
# B-Spline Volatility Surface
# ============================================================================

def create_bspline_vol_surface(moneyness_grid, T_grid, degree=3):
    """Create B-spline specification for volatility surface."""
    n_m = len(moneyness_grid) + degree - 1
    n_T = len(T_grid) + degree - 1

    # Create knot vectors with proper padding
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
# Calibration
# ============================================================================

def calibrate_vol_surface(df, S0, bspline_spec):
    """Calibrate volatility surface to market data."""
    n_m, n_T = bspline_spec['n_m'], bspline_spec['n_T']
    n_params = n_m * n_T

    # Initial guess: constant volatility = mean IV
    mean_iv = df['iv'].mean()
    x0 = np.full(n_params, mean_iv)

    # Market data
    m_data = df['moneyness'].values
    T_data = df['T'].values
    iv_data = df['iv'].values
    n_data = len(df)

    print(f"\nCalibration setup:")
    print(f"  Parameters: {n_params} ({n_m} x {n_T})")
    print(f"  Data points: {n_data}")
    print(f"  Mean IV: {mean_iv:.4f}")

    # Precompute basis functions for all data points
    knots_m = bspline_spec['knots_m']
    knots_T = bspline_spec['knots_T']
    degree = bspline_spec['degree']

    # Basis matrices: B_m[i, k] = Bₖ(mᵢ), B_T[i, j] = Bⱼ(Tᵢ)
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

    def objective(x):
        """Sum of squared errors."""
        coeffs = x.reshape(n_m, n_T)
        # σ_model = Σₖ Σⱼ w[k,j] * B_m[i,k] * B_T[i,j]
        sigma_model = np.einsum('ik,ij,kj->i', B_m_mat, B_T_mat, coeffs)
        errors = sigma_model - iv_data
        return 0.5 * np.sum(errors ** 2)

    def gradient(x):
        """Gradient of objective."""
        coeffs = x.reshape(n_m, n_T)
        sigma_model = np.einsum('ik,ij,kj->i', B_m_mat, B_T_mat, coeffs)
        errors = sigma_model - iv_data  # (n_data,)

        # ∂L/∂w[k,j] = Σᵢ errorᵢ * B_m[i,k] * B_T[i,j]
        grad = np.einsum('i,ik,ij->kj', errors, B_m_mat, B_T_mat)
        return grad.flatten()

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

    # Compute numerical Hessian at optimum
    print("\nComputing numerical Hessian (for comparison)...")
    t0 = time.time()
    eps = 1e-5
    n = len(result.x)
    H_num = np.zeros((n, n))

    g0 = gradient(result.x)
    for i in range(n):
        x_plus = result.x.copy()
        x_plus[i] += eps
        g_plus = gradient(x_plus)
        H_num[i, :] = (g_plus - g0) / eps

    # Symmetrize
    H_num = 0.5 * (H_num + H_num.T)
    t_hess = time.time() - t0

    print(f"Numerical Hessian time: {t_hess:.2f}s")
    print(f"Hessian shape: {H_num.shape}")
    print(f"Hessian condition number: {np.linalg.cond(H_num):.2e}")

    return {
        'coeffs': result.x.reshape(n_m, n_T),
        'result': result,
        'hessian': H_num,
        'bspline_spec': bspline_spec,
        't_opt': t_opt,
        'B_m_mat': B_m_mat,
        'B_T_mat': B_T_mat,
        'iv_data': iv_data
    }


# ============================================================================
# Visualization
# ============================================================================

def plot_vol_surface(calib_result, df, S0, save_path=None):
    """Plot the calibrated volatility surface."""
    coeffs = calib_result['coeffs']
    bspline_spec = calib_result['bspline_spec']

    # Create grid for plotting
    m_range = np.linspace(0.85, 1.15, 50)
    T_range = np.linspace(0.02, 0.8, 50)
    M, T = np.meshgrid(m_range, T_range)

    # Evaluate surface
    Z = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            Z[i, j] = eval_bspline_surface(M[i, j], T[i, j], coeffs, bspline_spec)

    # Plot
    fig = plt.figure(figsize=(16, 5))

    # 3D surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(M, T, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_xlabel('Moneyness (K/S)')
    ax1.set_ylabel('Time to Expiry (T)')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('B-Spline Volatility Surface')
    ax1.view_init(elev=25, azim=-60)
    fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.1)

    # Contour plot
    ax2 = fig.add_subplot(132)
    cs = ax2.contourf(M, T, Z, levels=20, cmap='viridis')
    ax2.scatter(df['moneyness'], df['T'], c=df['iv'], s=10, cmap='viridis',
                edgecolors='white', linewidth=0.3)
    ax2.set_xlabel('Moneyness (K/S)')
    ax2.set_ylabel('Time to Expiry (T)')
    ax2.set_title('Volatility Contours + Market Data')
    fig.colorbar(cs, ax=ax2)

    # Hessian heatmap
    if calib_result['hessian'] is not None:
        ax3 = fig.add_subplot(133)
        H = calib_result['hessian']
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


def plot_smile_comparison(calib_result, df, S0, T_slices=[0.05, 0.1, 0.25, 0.5], save_path=None):
    """Plot volatility smiles at different maturities."""
    coeffs = calib_result['coeffs']
    bspline_spec = calib_result['bspline_spec']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    m_range = np.linspace(0.85, 1.15, 100)

    for idx, T_target in enumerate(T_slices):
        ax = axes[idx]

        # Model smile
        sigma_model = [eval_bspline_surface(m, T_target, coeffs, bspline_spec) for m in m_range]
        ax.plot(m_range, sigma_model, 'b-', linewidth=2, label='B-Spline Model')

        # Market data near this T
        T_tol = 0.03
        mask = (df['T'] > T_target - T_tol) & (df['T'] < T_target + T_tol)
        df_slice = df[mask]

        if len(df_slice) > 0:
            ax.scatter(df_slice['moneyness'], df_slice['iv'], c='red', s=20,
                      label=f'Market (T≈{T_target:.2f})', zorder=5)

        ax.set_xlabel('Moneyness (K/S)')
        ax.set_ylabel('Implied Volatility')
        ax.set_title(f'Volatility Smile at T = {T_target:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.85, 1.15)

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
    print("FAST VOLATILITY SURFACE CALIBRATION")
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

    # Filter to reasonable moneyness range
    df = df[(df['moneyness'] > 0.8) & (df['moneyness'] < 1.2)]
    df = df[df['T'] < 1.0]  # Up to 1 year
    print(f"   Filtered (0.8 < m < 1.2, T < 1): {len(df):,}")

    # IV statistics
    print(f"\n   IV Statistics:")
    print(f"     Mean: {df['iv'].mean():.4f}")
    print(f"     Std:  {df['iv'].std():.4f}")
    print(f"     Min:  {df['iv'].min():.4f}")
    print(f"     Max:  {df['iv'].max():.4f}")

    # Create B-spline specification (smaller grid for speed)
    print("\n3. Setting up B-spline surface...")
    moneyness_grid = np.array([0.85, 0.95, 1.0, 1.05, 1.15])
    T_grid = np.array([0.02, 0.1, 0.25, 0.5])
    bspline_spec = create_bspline_vol_surface(moneyness_grid, T_grid, degree=3)
    print(f"   Grid: {len(moneyness_grid)} x {len(T_grid)}")
    print(f"   Parameters: {bspline_spec['n_m']} x {bspline_spec['n_T']} = {bspline_spec['n_m'] * bspline_spec['n_T']}")

    # Calibrate
    print("\n4. Calibrating volatility surface...")
    calib_result = calibrate_vol_surface(df, S0, bspline_spec)

    # Visualize
    print("\n5. Generating visualizations...")
    plot_vol_surface(calib_result, df, S0, save_path='vol_surface.png')
    plot_smile_comparison(calib_result, df, S0, save_path='vol_smiles.png')

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)

    # Summary
    print("\nSummary:")
    print(f"  Calibrated {bspline_spec['n_m'] * bspline_spec['n_T']} parameters")
    print(f"  RMSE: {np.sqrt(2*calib_result['result'].fun/len(df)):.4f}")
    print(f"  Optimization time: {calib_result['t_opt']:.2f}s")
    print(f"\nFigures saved:")
    print(f"  - vol_surface.png")
    print(f"  - vol_smiles.png")

    return calib_result, df, S0


if __name__ == "__main__":
    result, df, S0 = main()
