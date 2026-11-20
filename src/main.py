"""
Author: Jake Vick
Purpose: End-to-end execution: download data, compute log returns, fit distributions, plot,
print VaR & CVaR for the selected --methods (Normal / Student-t by default; Historical and the
PyMC Bayesian Student-t on request), and (with --backtest) run the rolling VaR backtest

Run w/ python -m src.main --coin_symbol eth
       python -m src.main --coin_symbol btc --methods normal,t,hist,bayes
       python -m src.main --coin_symbol btc --methods normal,t,bayes --backtest --bayes_engine conjugate
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
from .backtest import rolling_backtest, save_backtest_outputs
from .rolling_bayes import ENGINES, BT_DRAWS, BT_TUNE, BT_CHAINS
from .timing import (
    STAGE_BAYES,
    STAGE_BACKTEST,
    STAGE_DATA_PULL,
    STAGE_FITS,
    STAGE_PLOTS,
    STAGE_RETURNS,
    STAGE_RISK_METRICS,
    STAGE_ROLLING_VAR_PLOT,
    configure_logging,
    get_collector,
    new_run_id,
    reset_collector,
    stage,
)

METHOD_CHOICES = ("normal", "t", "hist", "bayes")
DEFAULT_METHODS = "normal,t"  # current behavior; bayes is opt-in because MCMC is slow
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

def parse_methods(value: str) -> list:
    """Parse a comma-separated --methods string, e.g. 'normal,t,hist,bayes'."""
    methods = [m.strip().lower() for m in value.split(",") if m.strip()]
    bad = [m for m in methods if m not in METHOD_CHOICES]
    if bad or not methods:
        raise argparse.ArgumentTypeError(
            f"invalid --methods {value!r}; choose from {','.join(METHOD_CHOICES)}"
        )
    return list(dict.fromkeys(methods))  # de-duplicate, keep order

def parse_log_level(value: str) -> str:
    v = value.upper()
    if v not in LOG_LEVELS:
        raise argparse.ArgumentTypeError(
            f"invalid --log-level {value!r}; choose from {', '.join(LOG_LEVELS)}"
        )
    return v

def etl(coin_symbol: str, vs_currency: str, base_dir: Path, on_stage=None):
    """Runs full ETL: Fetch -> save raw -> compute log returns -> save processed"""
    if on_stage is not None and get_collector().on_stage is None:
        get_collector().on_stage = on_stage

    raw_path = base_dir / "data" / "raw" / f"{coin_symbol}-{vs_currency}-max.csv"
    returns_path = base_dir / "data" / "output" / f"{coin_symbol}-log-returns.csv"

    with stage(STAGE_DATA_PULL) as st:
        print(f"Fetching raw data for {coin_symbol.upper()}-{vs_currency.upper()} from Yahoo Finance...")
        ticker = f"{coin_symbol.upper()}-{vs_currency.upper()}"
        df_raw = fetch_historical_data(ticker)
        save_raw_data(df_raw, raw_path)
        print(f"Raw data for {coin_symbol} downloaded.")
        st.payload = df_raw.head()

    with stage(STAGE_RETURNS) as st:
        print(f"Computing log returns for {raw_path}...")
        compute_log_returns(
            input_path=raw_path,
            output_path=returns_path,
            price_col="price",
            date_col="snapped_at",
            verbose=True
        )
        st.payload = pd.read_csv(returns_path)["return"]

    return returns_path

def main(alpha: float = 0.05, coin_symbol: str = "btc", vs_currency: str = "usd", backtest: bool = False, window_size: int = 252,
         methods: list = None, seed: int = 42, draws: int = 1000, tune: int = 1000, chains: int = 4, hdi_prob: float = 0.94,
         bayes_engine: str = "mcmc", refit_every: int = None, bt_draws: int = BT_DRAWS, bt_tune: int = BT_TUNE,
         bt_chains: int = BT_CHAINS, use_cache: bool = True, on_stage=None, profile: bool = False, no_plots: bool = False,
         log_level: str = None):
    if log_level:
        configure_logging(log_level)
    collector = reset_collector(on_stage=on_stage)
    if methods is None:
        methods = parse_methods(DEFAULT_METHODS)
    base_dir = Path(__file__).resolve().parents[1]

    returns_path = etl(coin_symbol, vs_currency, base_dir, on_stage=None)

    df_returns = pd.read_csv(returns_path)
    returns = df_returns["return"]

    print(f"\nLoaded {len(returns)} log returns for {coin_symbol.upper()}-{vs_currency.upper()} analysis")

    with stage(STAGE_FITS):
        normal_params = fit_normal(returns)
        t_params = fit_student_t(returns)

    if not no_plots:
        with stage(STAGE_PLOTS) as st:
            st.payload = [
                plot_histogram_with_fits(returns, normal_params, t_params, alpha),
                plot_qq_plots(returns, normal_params, t_params),
                plot_tail_comparison(normal_params, t_params),
            ]

    mu, sigma = normal_params
    df, loc, scale = t_params

    with stage(STAGE_RISK_METRICS) as st:
        print(f"\nRisk Metrics ({int(alpha*100)}%) for {coin_symbol.upper()}-{vs_currency.upper()}\n")
        metrics = {}
        if "normal" in methods:
            metrics["normal_var"] = compute_var_normal(mu, sigma, alpha)
            metrics["normal_cvar"] = compute_cvar_normal(mu, sigma, alpha)
            print(f"Normal VaR:  {metrics['normal_var']:.4f}")
            print(f"Normal CVaR: {metrics['normal_cvar']:.4f}\n")

        if "t" in methods:
            metrics["student_t_var"] = compute_var_student_t(df, loc, scale, alpha)
            metrics["student_t_cvar"] = compute_cvar_student_t(df, loc, scale, alpha)
            print(f"Student-t VaR:  {metrics['student_t_var']:.4f}")
            print(f"Student-t CVaR: {metrics['student_t_cvar']:.4f}")
            print(f"Student-t df = {df:.4f}, loc = {loc:.6f}, scale = {scale:.6f}\n")

        if "hist" in methods:
            metrics["historical_var"] = compute_var_historical(returns, alpha)
            metrics["historical_cvar"] = compute_cvar_historical(returns, alpha)
            print(f"Historical VaR:  {metrics['historical_var']:.4f}")
            print(f"Historical CVaR: {metrics['historical_cvar']:.4f}\n")
        st.payload = metrics

    if "bayes" in methods:
        from .bayes import run_bayes, print_bayes_summary  # lazy: PyMC import is slow

        with stage(STAGE_BAYES) as st:
            print(f"Running Bayesian Student-t (PyMC NUTS: {chains} chains x {draws} draws, tune {tune}, seed {seed})...")
            bayes_summary, bayes_runtime = run_bayes(
                returns, alpha=alpha, draws=draws, tune=tune, chains=chains, seed=seed, hdi_prob=hdi_prob
            )
            print_bayes_summary(bayes_summary, bayes_runtime)

            bayes_path = base_dir / "data" / "output" / f"{coin_symbol}-bayes-summary.csv"
            bayes_path.parent.mkdir(parents=True, exist_ok=True)
            bayes_summary.to_csv(bayes_path)
            print(f"Bayes summary saved to {bayes_path}")
            st.payload = bayes_summary

    if backtest:
        bt_methods = ["normal", "student_t", "historical"] + (["bayes"] if "bayes" in methods else [])
        print(f"\nRunning rolling backtest with window={window_size}...")
        df_returns["date"] = pd.to_datetime(df_returns["date"])  # Ensure datetime
        with stage(STAGE_BACKTEST) as st:
            results_df, summary_df, refits_df = rolling_backtest(
                df_returns, window_size, alpha, methods=bt_methods,
                bayes_engine=bayes_engine, refit_every=refit_every, bayes_draws=bt_draws,
                bayes_tune=bt_tune, bayes_chains=bt_chains, seed=seed,
                cache_dir=base_dir / "data" / "cache" / "rolling-bayes" if use_cache else None,
                return_refits=True,
                on_stage=None,
            )
            st.payload = results_df.head()

        print("\nBacktest Results (first 5 rows):")
        print(results_df.head().to_string(index=False))

        print("\nBacktest Summary (Kupiec POF, Christoffersen independence & conditional coverage):")
        print(summary_df.to_string())

        save_backtest_outputs(base_dir, coin_symbol, results_df, summary_df, refits_df)

        if not no_plots:
            with stage(STAGE_ROLLING_VAR_PLOT) as st:
                st.payload = plot_rolling_var(results_df, summary_df, methods=bt_methods, alpha=alpha, coin_symbol=coin_symbol)

    if no_plots:
        plt.close("all")
    else:
        plt.show()

    if profile:
        run_id = new_run_id()
        settings = {
            "coin_symbol": coin_symbol,
            "vs_currency": vs_currency,
            "alpha": alpha,
            "backtest": backtest,
            "window_size": window_size,
            "methods": ",".join(methods),
            "seed": seed,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "hdi_prob": hdi_prob,
            "bayes_engine": bayes_engine,
            "refit_every": "" if refit_every is None else refit_every,
            "bt_draws": bt_draws,
            "bt_tune": bt_tune,
            "bt_chains": bt_chains,
            "use_cache": use_cache,
            "no_plots": no_plots,
        }
        timings_path = base_dir / "data" / "output" / f"{coin_symbol}-timings.csv"
        collector.to_csv(timings_path, run_id, settings)
        collector.print_summary(run_id=run_id)
        print(f"Timing summary saved to {timings_path}")

def build_parser() -> argparse.ArgumentParser:
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
        default=parse_methods(DEFAULT_METHODS),
        help="Comma-separated full-sample methods: normal, t, hist, bayes (default: normal,t; bayes is slow). "
             "--backtest always runs normal, student_t and historical, and adds the rolling Bayes backtest when bayes is listed"
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

    parser.add_argument(
        "--bayes_engine",
        choices=ENGINES,
        default="mcmc",
        help="Backtest Bayes engine: mcmc (Student-t, NUTS, default) or conjugate (Normal-Inverse-Gamma, closed form, fast)"
    )
    parser.add_argument(
        "--refit_every",
        type=int,
        default=None,
        help="Backtest Bayes: refit every k days, reusing the last posterior in between (default: 21 for mcmc, 1 for conjugate)"
    )
    parser.add_argument("--bt_draws", type=int, default=BT_DRAWS,
                        help=f"Backtest Bayes (mcmc): draws per chain per refit (default: {BT_DRAWS})")
    parser.add_argument("--bt_tune", type=int, default=BT_TUNE,
                        help=f"Backtest Bayes (mcmc): tuning steps per chain per refit (default: {BT_TUNE})")
    parser.add_argument("--bt_chains", type=int, default=BT_CHAINS,
                        help=f"Backtest Bayes (mcmc): chains per refit, run sequentially (default: {BT_CHAINS})")
    parser.add_argument("--no_cache", action="store_true", default=False,
                        help="Backtest Bayes (mcmc): do not read or write cached fits in data/cache/rolling-bayes")
    parser.add_argument(
        "--log-level",
        dest="log_level",
        type=parse_log_level,
        default="WARNING",
        help="Python logging level (default: WARNING; INFO shows per-stage timing lines)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Print a stage timing table and write data/output/{coin_symbol}-timings.csv",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        default=False,
        help="Skip plot generation and plt.show() (for timing runs and non-interactive use)",
    )
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    main(alpha=args.alpha, coin_symbol=args.coin_symbol, vs_currency=args.vs_currency, backtest=args.backtest, window_size=args.window_size,
         methods=args.methods, seed=args.seed, draws=args.draws, tune=args.tune, chains=args.chains, hdi_prob=args.hdi_prob,
         bayes_engine=args.bayes_engine, refit_every=args.refit_every, bt_draws=args.bt_draws, bt_tune=args.bt_tune,
         bt_chains=args.bt_chains, use_cache=not args.no_cache, profile=args.profile, no_plots=args.no_plots,
         log_level=None)
