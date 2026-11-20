"""
Author: Jake Vick
Purpose: Rolling VaR est w/ out of sample backtesting and summary stats
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import multiprocessing as mp
import os
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from .backtest_tests import christoffersen_cc
from .distributions import fit_normal, fit_student_t
from .risk_metrics import (
    compute_var_normal, compute_cvar_normal,
    compute_var_student_t, compute_cvar_student_t,
    compute_var_historical, compute_cvar_historical
)
from .rolling_bayes import BT_CHAINS, BT_DRAWS, BT_TUNE
from .timing import STAGE_BACKTEST_LOOP, STAGE_BACKTEST_TESTS, STAGE_ROLLING_BAYES, get_collector, stage


def default_workers() -> int:
    """CLI default: min(4, cpu_count), at least 1."""
    return max(1, min(4, os.cpu_count() or 1))


def _chunk_ranges(window_size: int, n: int, workers: int) -> list:
    """Contiguous forecast-index ranges [start, end) covering i = window_size .. n-1."""
    n_windows = n - window_size
    if n_windows <= 0:
        return []
    n_chunks = min(n_windows, max(workers * 4, workers))
    bounds = np.linspace(0, n_windows, n_chunks + 1, dtype=int)
    ranges = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            ranges.append((int(window_size + a), int(window_size + b)))
    return ranges


def _evaluate_windows_chunk(payload):
    """Picklable worker: fit a contiguous block of rolling windows.

    payload is a plain tuple so it pickles cleanly under multiprocessing 'spawn'
    (the Windows start method). No lambdas, no closures.
    """
    (returns, start, end, window_size, alpha, methods,
     rolling_mean, rolling_std) = payload
    n_out = end - start
    out = {}
    if "normal" in methods:
        out["normal_var"] = np.empty(n_out, dtype=float)
        out["normal_cvar"] = np.empty(n_out, dtype=float)
    if "student_t" in methods:
        out["student_t_var"] = np.empty(n_out, dtype=float)
        out["student_t_cvar"] = np.empty(n_out, dtype=float)
    if "historical" in methods:
        out["historical_var"] = np.empty(n_out, dtype=float)
        out["historical_cvar"] = np.empty(n_out, dtype=float)

    for k, i in enumerate(range(start, end)):
        window_returns = returns[i - window_size:i]
        if "normal" in methods:
            mu = rolling_mean[i - 1]
            sigma = rolling_std[i - 1]
            out["normal_var"][k] = compute_var_normal(mu, sigma, alpha)
            out["normal_cvar"][k] = compute_cvar_normal(mu, sigma, alpha)
        if "student_t" in methods:
            df_t, loc, scale = fit_student_t(pd.Series(window_returns))
            out["student_t_var"][k] = compute_var_student_t(df_t, loc, scale, alpha)
            out["student_t_cvar"][k] = compute_cvar_student_t(df_t, loc, scale, alpha)
        if "historical" in methods:
            series = pd.Series(window_returns)
            out["historical_var"][k] = compute_var_historical(series, alpha)
            out["historical_cvar"][k] = compute_cvar_historical(series, alpha)
    return start, end, out


def _run_serial_loop(returns, window_size, alpha, methods, rolling_mean, rolling_std, results, t_warm_start: bool):
    """Original one-window-at-a-time loop. --workers 1 always takes this path."""
    n = len(returns)
    prev_t = None
    for i in tqdm(range(window_size, n), desc="Rolling Backtest"):
        window_returns = returns[i - window_size:i]
        idx = i - window_size
        if "normal" in methods:
            mu = rolling_mean[i - 1]
            sigma = rolling_std[i - 1]
            results["normal_var"][idx] = compute_var_normal(mu, sigma, alpha)
            results["normal_cvar"][idx] = compute_cvar_normal(mu, sigma, alpha)
        if "student_t" in methods:
            start = prev_t if t_warm_start else None
            df_t, loc, scale = fit_student_t(pd.Series(window_returns), start=start)
            if t_warm_start:
                prev_t = (df_t, loc, scale)
            results["student_t_var"][idx] = compute_var_student_t(df_t, loc, scale, alpha)
            results["student_t_cvar"][idx] = compute_cvar_student_t(df_t, loc, scale, alpha)
        if "historical" in methods:
            series = pd.Series(window_returns)
            results["historical_var"][idx] = compute_var_historical(series, alpha)
            results["historical_cvar"][idx] = compute_cvar_historical(series, alpha)


def _run_parallel_loop(returns, window_size, alpha, methods, rolling_mean, rolling_std, results, workers: int):
    """Chunked ProcessPoolExecutor with the spawn start method (Windows-safe)."""
    n = len(returns)
    ranges = _chunk_ranges(window_size, n, workers)
    methods_t = tuple(methods)
    payload_common = (np.asarray(returns, dtype=float), window_size, float(alpha), methods_t,
                      np.asarray(rolling_mean, dtype=float), np.asarray(rolling_std, dtype=float))
    ctx = mp.get_context("spawn")
    futures = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for start, end in ranges:
            payload = (payload_common[0], start, end, payload_common[1], payload_common[2],
                       payload_common[3], payload_common[4], payload_common[5])
            futures.append(pool.submit(_evaluate_windows_chunk, payload))
        with tqdm(total=n - window_size, desc=f"Rolling Backtest ({workers} workers)") as pbar:
            for fut in as_completed(futures):
                start, end, out = fut.result()
                sl = slice(start - window_size, end - window_size)
                for key, arr in out.items():
                    results[key][sl] = arr
                pbar.update(end - start)


def rolling_backtest(df_returns: pd.DataFrame, window_size: int = 252, alpha: float = 0.05, methods: list = ["normal", "student_t", "historical"],
                     bayes_engine: str = "mcmc", refit_every: Optional[int] = None, bayes_draws: int = BT_DRAWS,
                     bayes_tune: int = BT_TUNE, bayes_chains: int = BT_CHAINS, seed: int = 42,
                     cache_dir: Optional[Path] = None, return_refits: bool = False, on_stage=None,
                     workers: int = 1, t_warm_start: bool = False):
    """
    Perform rolling window VaR estimation and out-of-sample backtest.

    Parameters ----------
    df_returns, window_size, alpha and methods (normal, student_t, historical, bayes)
    bayes only (see src/rolling_bayes.py): bayes_engine ("mcmc" or "conjugate"), refit_every
    (k days between refits; default 21 for mcmc, 1 for conjugate), bayes_draws / bayes_tune /
    bayes_chains (NUTS per refit), seed (refit on day r uses seed + r), cache_dir (cached MCMC
    fits; None = no cache), return_refits (also return the per-refit table),
    on_stage (optional progress hook; attached to the run collector if it has none),
    workers (process count for the hist/normal/t loop; 1 = serial path, the default
    for library calls and tests; the CLI default is min(4, cpu_count)),
    t_warm_start (seed each Student-t fit with the previous window's (df, loc, scale);
    default off so outputs stay identical; forces the serial path)

    Output -------
    results_df: DataFrame with dates, actual returns, VaR/CVaR for each method, and breach flags
    summary_df: Per-method breaches, expected, breach rate, Kupiec POF / Christoffersen
                independence / conditional coverage LR stats, p-values, reject at 5%
    refits_df (only if return_refits): one row per Bayes refit (None if bayes not run)
    """
    if on_stage is not None and get_collector().on_stage is None:
        get_collector().on_stage = on_stage

    # Ensure sorted by date ascending
    df_returns = df_returns.sort_values("date").reset_index(drop=True)
    returns = df_returns["return"].values
    dates = pd.to_datetime(df_returns["date"])
    
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

    # Roll through data (Normal / Student-t / Historical). Precompute Normal rolling stats
    # inside this stage so it covers the hist/normal/t work.
    if workers < 1:
        raise ValueError(f"workers must be >= 1 (got {workers})")
    if t_warm_start and workers > 1:
        warnings.warn(
            "t_warm_start is sequential (each window seeds the next); using the serial path",
            RuntimeWarning,
        )
        workers = 1

    with stage(STAGE_BACKTEST_LOOP):
        returns_series = pd.Series(returns)
        rolling_mean = returns_series.rolling(window_size).mean().to_numpy()
        rolling_std = returns_series.rolling(window_size).std().to_numpy()
        if workers == 1:
            _run_serial_loop(returns, window_size, alpha, methods, rolling_mean, rolling_std, results, t_warm_start)
        else:
            _run_parallel_loop(returns, window_size, alpha, methods, rolling_mean, rolling_std, results, workers)

    # Bayes: refit every k days on returns[r - window_size : r] (see src/rolling_bayes.py).
    # Runs after the loop above, so the other methods are computed exactly as before.
    refits_df = None
    if "bayes" in methods:
        from .rolling_bayes import rolling_bayes, refit_report
        with stage(STAGE_ROLLING_BAYES):
            var_b, cvar_b, refits_df = rolling_bayes(
                returns, window_size=window_size, alpha=alpha, engine=bayes_engine,
                refit_every=refit_every, draws=bayes_draws, tune=bayes_tune, chains=bayes_chains,
                seed=seed, cache_dir=cache_dir,
            )
            results["bayes_var"] = var_b
            results["bayes_cvar"] = cvar_b
            date_str = dates.dt.strftime("%Y-%m-%d").to_numpy()
            refits_df.insert(2, "refit_date", date_str[refits_df["refit_pos"].to_numpy()])
            refits_df.insert(3, "fit_start_date", date_str[refits_df["fit_start_pos"].to_numpy()])
            refits_df.insert(4, "fit_end_date", date_str[refits_df["fit_end_pos"].to_numpy()])
            print(refit_report(refits_df))

    results_df = pd.DataFrame(results)

    # Flag breaches (actual < VaR, since VaR is negative for losses)
    for method in methods:
        results_df[f"{method}_breach"] = results_df["actual_return"] < results_df[f"{method}_var"]

    # Summary: breach rates plus Kupiec / Christoffersen tests.
    # Note: rolling windows overlap, so the tests' independence assumption is approximate.
    with stage(STAGE_BACKTEST_TESTS):
        summary = {m: christoffersen_cc(results_df[f"{m}_breach"].values, alpha) for m in methods}
        summary_df = pd.DataFrame(summary).T
        summary_df.index.name = "method"

    if return_refits:
        return results_df, summary_df, refits_df
    return results_df, summary_df

def save_backtest_outputs(base_dir: Path, coin_symbol: str, results_df: pd.DataFrame, summary_df: pd.DataFrame,
                          refits_df: Optional[pd.DataFrame] = None) -> None:
    """Write data/backtest/{coin}-backtest-results.csv, -summary.csv and (Bayes) -bayes-refits.csv."""
    out_dir = base_dir / "data" / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{coin_symbol}-backtest-results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Full backtest results saved to {results_path}")
    summary_path = out_dir / f"{coin_symbol}-backtest-summary.csv"
    summary_df.to_csv(summary_path)
    print(f"Backtest summary saved to {summary_path}")
    if refits_df is not None:
        refits_path = out_dir / f"{coin_symbol}-bayes-refits.csv"
        refits_df.to_csv(refits_path, index=False)
        print(f"Bayes refit log saved to {refits_path}")


BACKTEST_METHODS = ("normal", "student_t", "historical", "bayes")


def parse_backtest_methods(value: str) -> list:
    """Comma-separated backtest methods, e.g. 'normal,student_t,historical,bayes'."""
    methods = [m.strip().lower() for m in value.split(",") if m.strip()]
    bad = [m for m in methods if m not in BACKTEST_METHODS]
    if bad or not methods:
        raise ValueError(f"invalid methods {value!r}; choose from {','.join(BACKTEST_METHODS)}")
    return list(dict.fromkeys(methods))


if __name__ == "__main__":
    """
    Runs on the existing returns CSV (no download). Example:
        python -m src.backtest --coin_symbol btc --methods normal,student_t,historical,bayes \
            --bayes_engine conjugate --save
    """
    import argparse
    import time
    from .rolling_bayes import ENGINES

    parser = argparse.ArgumentParser(description="Backtest VaR models on data/output/{coin}-log-returns.csv")
    parser.add_argument("--coin_symbol", type=str, default="btc")
    parser.add_argument("--vs_currency", type=str, default="usd")
    parser.add_argument("--window_size", type=int, default=252)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--methods", type=parse_backtest_methods, default="normal,student_t,historical",
                        help="Comma-separated: normal, student_t, historical, bayes (default: the first three)")
    parser.add_argument("--bayes_engine", choices=ENGINES, default="mcmc")
    parser.add_argument("--refit_every", type=int, default=None, help="Bayes refit interval k (default 21 mcmc, 1 conjugate)")
    parser.add_argument("--bt_draws", type=int, default=BT_DRAWS)
    parser.add_argument("--bt_tune", type=int, default=BT_TUNE)
    parser.add_argument("--bt_chains", type=int, default=BT_CHAINS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cache", action="store_true", help="Do not read or write data/cache/rolling-bayes")
    parser.add_argument("--save", action="store_true", help="Write the results / summary / refit CSVs to data/backtest")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help=f"Process count for the rolling hist/normal/t loop (default: min(4, cpu_count)={default_workers()}; 1 = serial)")
    parser.add_argument("--t_warm_start", action="store_true", default=False,
                        help="Seed each Student-t fit with the previous window's (df, loc, scale); serial; default off")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    returns_path = base_dir / "data" / "output" / f"{args.coin_symbol}-log-returns.csv"

    df_returns = pd.read_csv(returns_path)
    df_returns["date"] = pd.to_datetime(df_returns["date"])

    t0 = time.perf_counter()
    results_df, summary_df, refits_df = rolling_backtest(
        df_returns, args.window_size, args.alpha, methods=args.methods,
        bayes_engine=args.bayes_engine, refit_every=args.refit_every, bayes_draws=args.bt_draws,
        bayes_tune=args.bt_tune, bayes_chains=args.bt_chains, seed=args.seed,
        cache_dir=None if args.no_cache else base_dir / "data" / "cache" / "rolling-bayes",
        return_refits=True,
        workers=args.workers,
        t_warm_start=args.t_warm_start,
    )
    print(f"Backtest took {time.perf_counter() - t0:.1f} s")

    print("\nBacktest Results Sample:")
    print(results_df.head())

    print("\nBreach Rate Summary:")
    print(summary_df)

    if args.save:
        save_backtest_outputs(base_dir, args.coin_symbol, results_df, summary_df, refits_df)
