"""
Author: Jake Vick
Purpose: Compute Normal & Student-t distributions for VaR / CVaR modeling
"""

from typing import Tuple
from scipy.stats import norm, t
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd

def fit_normal(returns: pd.Series) -> Tuple[float, float]:
    """Fit a Normal distribution to returns using MLE.
    
    Parameters ----------
    returns: Series of asset returns.
    
    Output ----------
    mu, sigma: Mean and standard deviation.
    """

    mu = np.mean(returns)
    sigma = np.std(returns)
    return mu, sigma

DF_FLOOR = 2.01
"""Lower bound on Student-t degrees of freedom (variance is finite only for df > 2)."""

DF_NORMAL_FALLBACK = 1e6
"""df used when the t fit fails; a t with this df is numerically Normal."""


def _valid(df: float, loc: float, scale: float) -> bool:
    return all(np.isfinite([df, loc, scale])) and df > 0 and scale > 0


def fit_student_t(returns: pd.Series) -> Tuple[float, float, float]:
    """Fit a Student's t distribution to returns using MLE, with df bounded to df >= DF_FLOOR.

    If the unconstrained MLE gives df < DF_FLOOR, the fit is redone with df fixed at
    DF_FLOOR. If the fit fails or returns invalid parameters, it falls back to a
    Normal-equivalent t (df = DF_NORMAL_FALLBACK, loc = mean, scale = std) and warns.

    Parameters ----------
    returns: Series of asset returns.

    Output ----------
    df, loc, scale: Degrees of freedom (df), location, and scale. Note scale is NOT the
    standard deviation; for df > 2, std = scale * sqrt(df / (df - 2)).
    """

    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("fit_student_t: no finite returns to fit")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df, loc, scale = t.fit(x)
            if _valid(df, loc, scale) and df < DF_FLOOR:
                df, loc, scale = t.fit(x, f0=DF_FLOOR)
                df = DF_FLOOR
        if _valid(df, loc, scale) and df >= DF_FLOOR:
            return float(df), float(loc), float(scale)
        reason = f"invalid params (df={df}, loc={loc}, scale={scale})"
    except Exception as e:  # scipy optimiser failures
        reason = f"{type(e).__name__}: {e}"

    mu, sigma = fit_normal(x)
    scale = sigma if sigma > 0 else np.finfo(float).eps
    warnings.warn(f"Student-t fit failed ({reason}); falling back to Normal", RuntimeWarning)
    return DF_NORMAL_FALLBACK, float(mu), float(scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fitting Normal & Student-t distributions")
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
        print(f"Loaded {len(returns)} returns (mean = {returns.mean():.6f}, std = {returns.std():.6f})") #:.2f trims output to #.##

        mu, sig = fit_normal(returns)
        df, loc, scale = fit_student_t(returns)

        print(f"\nNormal: mu = {mu:.6f}, sig = {sig:.6f}")
        print(f"\nStudent-t: df = {df:.6f}, loc = {loc:.6f}, scale = {scale:.6f}")

    except FileNotFoundError:
        print("No file found...")
    except Exception as e:
        print(f"Error during test: {e}")