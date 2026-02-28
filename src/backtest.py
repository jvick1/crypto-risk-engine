"""
Author: Jake Vick
Purpose: Rolling VaR est w/ out of sample backtesting and summary stats
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

from .distributions import fit_normal, fit_student_t
from .risk_metrics import (
    compute_var_normal, compute_cvar_normal,
    compute_var_student_t, compute_cvar_student_t,
    compute_var_historical, compute_cvar_historical
)

def rolling_backtest(df_returns: pd.DataFrame, window_size: int = 252, alpha: float = 0.05, methods: list = ["normal", "student_t", "historical"]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform rolling window VaR estimation and out-of-sample backtest.

    Parameters ----------
    df_returns, window_size, alpha and methods (normal, student_t, historical)

    Output -------
    results_df: DataFrame with dates, actual returns, VaR/CVaR for each method, and breach flags
    summary_df: Breach rate summary per method
    """
    # Ensure sorted by date ascending
    df_returns = df_returns.sort_values("date").reset_index(drop=True)
    returns = df_returns["return"].values
    dates = pd.to_datetime(df_returns["date"])
    
    # Precompute rolling statistics for Normal model
    returns_series = pd.Series(returns)
    rolling_mean = returns_series.rolling(window_size).mean()
    rolling_std = returns_series.rolling(window_size).std()

    # Initialize results
    n = len(returns)
    results = {
        "date": dates[window_size:],
        "actual_return": returns[window_size:]
    }
    for method in methods:
        results[f"{method}_var"] = np.full(n - window_size, np.nan)
        results[f"{method}_cvar"] = np.full(n - window_size, np.nan)
        results[f"{method}_breach"] = np.full(n - window_size, False)

    # Roll through data
    for i in tqdm(range(window_size, n), desc="Rolling Backtest"):     
        window_returns = returns[i - window_size:i]  # Past window for fitting

        if "normal" in methods:
            mu = rolling_mean.iloc[i - 1]
            sigma = rolling_std.iloc[i - 1]
            results["normal_var"][i - window_size] = compute_var_normal(mu, sigma, alpha)
            results["normal_cvar"][i - window_size] = compute_cvar_normal(mu, sigma, alpha)

        if "student_t" in methods:
            df_t, loc, scale = fit_student_t(pd.Series(window_returns))
            results["student_t_var"][i - window_size] = compute_var_student_t(df_t, loc, scale, alpha)
            results["student_t_cvar"][i - window_size] = compute_cvar_student_t(df_t, loc, scale, alpha)

        if "historical" in methods:
            results["historical_var"][i - window_size] = compute_var_historical(pd.Series(window_returns), alpha)
            results["historical_cvar"][i - window_size] = compute_cvar_historical(pd.Series(window_returns), alpha)

    results_df = pd.DataFrame(results)

    # Flag breaches (actual < VaR, since VaR is negative for losses)
    for method in methods:
        results_df[f"{method}_breach"] = results_df["actual_return"] < results_df[f"{method}_var"]

    # Summary: Breach rates
    summary = {}
    for method in methods:
        breaches = results_df[f"{method}_breach"].sum()
        total_oos = len(results_df)
        breach_rate = breaches / total_oos if total_oos > 0 else 0
        summary[method] = {
            "breaches": breaches,
            "total_periods": total_oos,
            "breach_rate": breach_rate,
            "expected_rate": alpha
        }
    summary_df = pd.DataFrame(summary).T

    return results_df, summary_df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest VaR models")
    parser.add_argument("--coin_symbol", type=str, default="btc")
    parser.add_argument("--vs_currency", type=str, default="usd")
    parser.add_argument("--window_size", type=int, default=252)
    parser.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    returns_path = base_dir / "data" / "output" / f"{args.coin_symbol}-log-returns.csv"

    df_returns = pd.read_csv(returns_path)
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    results_df, summary_df = rolling_backtest(df_returns, args.window_size, args.alpha)

    print("\nBacktest Results Sample:")
    print(results_df.head())

    print("\nBreach Rate Summary:")
    print(summary_df)