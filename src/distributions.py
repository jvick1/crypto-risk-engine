"""
Author: Jake Vick
Purpose: Compute Normal & Student-t distributions for VaR / CVaR modeling
"""

from typing import Tuple
from scipy.stats import norm, t
from pathlib import Path
import argparse

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

def fit_student_t(returns: pd.Series) -> Tuple[float, float, float]:
    """Fit a Student's t distribution to returns using MLE.
    
    Parameters ----------
    returns: Series of asset returns.
    
    Output ----------
    df, loc, scale: Degrees of freedom (df), location (mean), scale (std).
    """

    df, loc, scale = t.fit(returns)
    return df, loc, scale

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