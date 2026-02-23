"""
Author: Jake Vick
Purpose: End-to-end execution of distribution fitting, visualization, and risk metrics

Run w/ python -m src.main --coin_symbol eth
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from .api import fetch_historical_data, save_raw_data
from .returns import compute_log_returns

from .distributions import fit_normal, fit_student_t
from .visualization import (
    plot_histogram_with_fits,
    plot_qq_plots,
    plot_tail_comparison
)
from .risk_metrics import (
    compute_var_normal,
    compute_cvar_normal,
    compute_var_student_t,
    compute_cvar_student_t
)

def etl(coin_symbol: str, vs_currency: str, base_dir: Path):
    """Runs full ETL: Fetch -> save raw -> compute log returns -> save processed"""
    
    raw_path = base_dir / "data" / "raw" / f"{coin_symbol}-{vs_currency}-max.csv"
    returns_path = base_dir / "data" / "output" / f"{coin_symbol}-log-returns.csv.csv"

    # API 
    print(f"Fetching raw data for {coin_symbol.upper()}-{vs_currency.upper()} from Yahoo Finance...")
    ticker = f"{coin_symbol.upper()}-{vs_currency.upper()}"
    df_raw = fetch_historical_data(ticker)
    save_raw_data(df_raw, raw_path)
    print(f"Raw data for {coin_symbol} downloaded.")

    # Compute Log Returns
    print(f"Computing log returns for {raw_path}...")
    compute_log_returns(
        input_path=raw_path,
        output_path=returns_path,
        price_col="price",
        date_col="snapped_at",
        verbose=True
    )

    return returns_path

def main(alpha: float = 0.05, coin_symbol: str = "btc", vs_currency: str = "usd"):
    base_dir = Path(__file__).resolve().parents[1]

    returns_path = etl(coin_symbol, vs_currency, base_dir)

    df_returns = pd.read_csv(returns_path)
    returns = df_returns["return"]

    print(f"\nLoaded {len(returns)} log returns for {coin_symbol.upper()}-{vs_currency.upper()} anaysis")

    # Fit distributions
    normal_params = fit_normal(returns)
    t_params = fit_student_t(returns)

    # Visual diagnostics
    plot_histogram_with_fits(returns, normal_params, t_params, alpha)
    plot_qq_plots(returns, normal_params, t_params)
    plot_tail_comparison(normal_params, t_params)

    # Risk metrics
    mu, sigma = normal_params
    df, loc, scale = t_params

    print(f"\nRisk Metrics ({int(alpha*100)}%) for {coin_symbol.upper()}-USD\n")

    print(f"Normal VaR:  {compute_var_normal(mu, sigma, alpha):.4f}")
    print(f"Normal CVaR: {compute_cvar_normal(mu, sigma, alpha):.4f}\n")

    print(f"Student-t VaR:  {compute_var_student_t(df, loc, scale, alpha):.4f}")
    print(f"Student-t CVaR: {compute_cvar_student_t(df, loc, scale, alpha):.4f}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto risk analysis: fit distributions & compute VaR/CVaR")
    parser.add_argument(
        "--coin_symbol",
        type=str,
        default="btc",
        help="Coin symbol (e.g., btc, eth, sol)"
    )
    parser.add_argument(
        "--vs_currency",
        type=str,
        default="usd",
        help="Quote currency (default: usd)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Confidence level for VaR/CVaR (default: 0.05 = 95%)"
    )

    args = parser.parse_args()

    main(alpha=args.alpha, coin_symbol=args.coin_symbol, vs_currency=args.vs_currency)