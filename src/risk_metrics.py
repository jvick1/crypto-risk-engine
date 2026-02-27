"""
Author: Jake Vick
Purpose: VaR / CVaR modeling
"""

from scipy.stats import norm, t
from pathlib import Path
import numpy as np
import argparse

import pandas as pd

def compute_var_normal(mu: float, sigma: float, alpha: float = 0.05) -> float:
    """
    Compute parametric VaR for Normal distribution (left-tail quantile).

    Parameters ----------
    mu: Mean return
    sigma: Standard deviation of returns
    alpha: Tail probability (0.05 for 95% VaR)

    Output ----------
    VaR estimate
    """
    return mu + sigma * norm.ppf(alpha)

def compute_cvar_normal(mu: float, sigma: float, alpha: float = 0.05) -> float:
    """
    Compute parametric CVaR (Expected Shortfall) for Normal distribution.

    Parameters ----------
    mu, sigma, alpha

    Output ----------
    CVaR estimate
    """
    z = norm.ppf(alpha)
    return mu - sigma * (norm.pdf(z) / alpha)

def compute_var_student_t(df: float, loc: float, scale: float, alpha: float = 0.05) -> float:
    """
    Compute parametric VaR for Student-t distribution.

    Parameters ----------
    df: Degrees of freedom
    loc: Location parameter
    scale: Scale parameter
    alpha: Tail probability

    Output ----------
    VaR estimate
    """
    return loc + scale * t.ppf(alpha, df)

def compute_cvar_student_t(df: float, loc: float, scale: float, alpha: float = 0.05) -> float:
    """
    Compute parametric CVaR (Expected Shortfall) for Student-t distribution.

    Parameters ----------
    df, loc, scale, alpha

    Output ----------
    CVaR estimate
    """
    t_q = t.ppf(alpha, df)
    density = t.pdf(t_q, df)
    adjustment = (df + t_q**2) / (df - 1)
    return loc - scale * (density / alpha) * adjustment

def compute_var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Compute historical (empirical) VaR as the alpha-quantile of returns.

    Parameters ----------
    returns, alpha

    Output -------
    Historical VaR estimate
    """
    return np.percentile(returns, alpha * 100)

def compute_cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Compute historical CVaR as the mean of returns below the VaR threshold.

    Parameters ----------
    returns, alpha

    Output -------
    Historical CVaR estimate
    """
    var = compute_var_historical(returns, alpha)
    return returns[returns <= var].mean()

if __name__ == "__main__":
    """
    Run with:
        python -m src.risk_metrics
    """

    from .distributions import fit_normal, fit_student_t

    parser = argparse.ArgumentParser(description="Risk Metrics for Normal & Student-t distributions")
    parser.add_argument(
        "--coin_symbol",
        type=str,
        default="btc",
        help="Coin symbol to load returns for (e.g., btc, eth, sol)"
    )
    parser.add_argument(
        "--vs_currency",
        type=str,
        default="usd",
        help="Quote currency (default: usd)"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    returns_path = base_dir / "data" / "output" / f"{args.coin_symbol}-log-returns.csv"

    try:
        returns = pd.read_csv(returns_path)["return"]

        # Fit distributions
        mu, sigma = fit_normal(returns)
        df, loc, scale = fit_student_t(returns)

        alpha = 0.05

        print("Risk Metric Test\n")

        print(f"Normal VaR (95%):  {compute_var_normal(mu, sigma, alpha):.4f}")
        print(f"Normal CVaR (95%): {compute_cvar_normal(mu, sigma, alpha):.4f}\n")

        print(f"Student-t VaR (95%):  {compute_var_student_t(df, loc, scale, alpha):.4f}")
        print(f"Student-t CVaR (95%): {compute_cvar_student_t(df, loc, scale, alpha):.4f}")

        print(f"Historical VaR (95%):  {compute_var_historical(returns, alpha):.4f}")
        print(f"Historical CVaR (95%): {compute_cvar_historical(returns, alpha):.4f}")

    except Exception as e:
        print(f"Risk metric test failed: {e}")