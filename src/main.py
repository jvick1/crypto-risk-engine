"""
Author: Jake Vick
Purpose: End-to-end execution: download data, compute log returns, fit distributions, plot,
print VaR & CVaR for the selected --methods (Normal / Student-t by default; Historical and the
PyMC Bayesian Student-t on request), and (with --backtest) run the rolling VaR backtest

Run w/ python -m src.main --coin_symbol eth
       python -m src.main --coin_symbol btc --methods normal,t,hist,bayes
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
    plot_tail_comparison,
    plot_rolling_var
)
from .risk_metrics import (
    compute_var_normal,
    compute_cvar_normal,
    compute_var_student_t,
    compute_cvar_student_t,
    compute_var_historical,
    compute_cvar_historical
)
from .backtest import rolling_backtest

METHOD_CHOICES = ("normal", "t", "hist", "bayes")
DEFAULT_METHODS = "normal,t"  # current behavior; bayes is opt-in because MCMC is slow

def parse_methods(value: str) -> list:
    """Parse a comma-separated --methods string, e.g. 'normal,t,hist,bayes'."""
    methods = [m.strip().lower() for m in value.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHOD_CHOICES]
    if bad or not methods:
        raise argparse.ArgumentTypeError(
            f"invalid --methods {value!r}; choose from {','.join(METHOD_CHOICES)}"
        )
    return list(dict.fromkeys(methods))  # de-duplicate, keep order

def etl(coin_symbol: str, vs_currency: str, base_dir: Path):
    """Runs full ETL: Fetch -> save raw -> compute log returns -> save processed"""
    
    raw_path = base_dir / "data" / "raw" / f"{coin_symbol}-{vs_currency}-max.csv"
    returns_path = base_dir / "data" / "output" / f"{coin_symbol}-log-returns.csv"

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

def main(alpha: float = 0.05, coin_symbol: str = "btc", vs_currency: str = "usd", backtest: bool = False, window_size: int = 252,
         methods: list = None, seed: int = 42, draws: int = 1000, tune: int = 1000, chains: int = 4, hdi_prob: float = 0.94):
    if methods is None:
        methods = parse_methods(DEFAULT_METHODS)
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

    if "normal" in methods:
        print(f"Normal VaR:  {compute_var_normal(mu, sigma, alpha):.4f}")
        print(f"Normal CVaR: {compute_cvar_normal(mu, sigma, alpha):.4f}\n")

    if "t" in methods:
        print(f"Student-t VaR:  {compute_var_student_t(df, loc, scale, alpha):.4f}")
        print(f"Student-t CVaR: {compute_cvar_student_t(df, loc, scale, alpha):.4f}")
        print(f"Student-t df = {df:.4f}, loc = {loc:.6f}, scale = {scale:.6f}\n")

    if "hist" in methods:
        print(f"Historical VaR:  {compute_var_historical(returns, alpha):.4f}")
        print(f"Historical CVaR: {compute_cvar_historical(returns, alpha):.4f}\n")

    if "bayes" in methods:
        from .bayes import run_bayes, print_bayes_summary  # lazy: PyMC import is slow

        print(f"Running Bayesian Student-t (PyMC NUTS: {chains} chains x {draws} draws, tune {tune}, seed {seed})...")
        bayes_summary, bayes_runtime = run_bayes(
            returns, alpha=alpha, draws=draws, tune=tune, chains=chains, seed=seed, hdi_prob=hdi_prob
        )
        print_bayes_summary(bayes_summary, bayes_runtime)

        bayes_path = base_dir / "data" / "output" / f"{coin_symbol}-bayes-summary.csv"
        bayes_path.parent.mkdir(parents=True, exist_ok=True)
        bayes_summary.to_csv(bayes_path)
        print(f"Bayes summary saved to {bayes_path}")

    if backtest:
        print(f"\nRunning rolling backtest with window={window_size}...")
        df_returns["date"] = pd.to_datetime(df_returns["date"])  # Ensure datetime
        results_df, summary_df = rolling_backtest(df_returns, window_size, alpha)

        print("\nBacktest Results (first 5 rows):")
        print(results_df.head().to_string(index=False))

        print("\nBacktest Summary (Kupiec POF, Christoffersen independence & conditional coverage):")
        print(summary_df.to_string())

        backtest_path = base_dir / "data" / "backtest" / f"{coin_symbol}-backtest-results.csv"
        backtest_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(backtest_path, index=False)
        print(f"Full backtest results saved to {backtest_path}")

        summary_path = base_dir / "data" / "backtest" / f"{coin_symbol}-backtest-summary.csv"
        summary_df.to_csv(summary_path)
        print(f"Backtest summary saved to {summary_path}")

        plot_rolling_var(results_df, summary_df, methods=["normal", "student_t", "historical"], alpha=alpha, coin_symbol=coin_symbol)

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
        help="Tail probability for VaR/CVaR (default: 0.05, i.e. 95%% VaR)"
    )
    parser.add_argument(
        "--backtest", 
        action="store_true", 
        default=False, 
        help="Run out-of-sample backtest"
    )
    parser.add_argument(
        "--window_size", 
        type=int, 
        default=252, 
        help="Rolling window size (days)"
    )

    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=DEFAULT_METHODS,
        help="Comma-separated full-sample methods: normal, t, hist, bayes (default: normal,t; bayes is slow). "
             "The --backtest methods are unchanged (normal, student_t, historical)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the Bayesian sampler and posterior predictive (default: 42)"
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="Bayes: posterior draws per chain (default: 1000)"
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Bayes: tuning (warm-up) steps per chain, discarded (default: 1000)"
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Bayes: MCMC chains, run in parallel up to the CPU count (default: 4)"
    )
    parser.add_argument(
        "--hdi_prob",
        type=float,
        default=0.94,
        help="Bayes: highest-density credible interval mass (default: 0.94)"
    )

    args = parser.parse_args()

    main(alpha=args.alpha, coin_symbol=args.coin_symbol, vs_currency=args.vs_currency, backtest=args.backtest, window_size=args.window_size,
         methods=args.methods, seed=args.seed, draws=args.draws, tune=args.tune, chains=args.chains, hdi_prob=args.hdi_prob)