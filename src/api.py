"""
Author: Jake Vick
Purpose: ETL extract layer to fetch raw historical price data from yfinance and save as CSV.
This feeds into returns.py for log return computation.
"""

from pathlib import Path
from typing import Optional
import argparse
import warnings

import pandas as pd
import yfinance as yf

try:
    from yfinance.exceptions import YFException, YFRateLimitError
except ImportError:  # older yfinance
    YFRateLimitError = type("YFRateLimitError", (Exception,), {})
    YFException = type("YFException", (Exception,), {})

DELTA_LOOKBACK_DAYS = 5


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance may return MultiIndex columns (Price, Ticker); keep the first level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _normalize_downloaded(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Date and Close, renamed to snapped_at (UTC) and price (8 dp)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["snapped_at", "price"])
    df = _flatten_columns(df)
    out = df[["Close"]].reset_index()
    out = _flatten_columns(out)
    out = out.rename(columns={"Date": "snapped_at", "Close": "price"})
    out["snapped_at"] = pd.to_datetime(out["snapped_at"], utc=True)
    out["price"] = pd.to_numeric(out["price"], errors="coerce").round(8)
    out = out.dropna(subset=["snapped_at", "price"])
    return out[["snapped_at", "price"]].reset_index(drop=True)


def fetch_historical_data(ticker: str, start=None) -> pd.DataFrame:
    """
    Fetch historical market data for a cryptocurrency using yfinance.

    Parameters
    ----------
    ticker: Yahoo Finance ticker symbol (e.g., 'BTC-USD', 'ETH-USD', 'SOL-USD')
    start: Optional start date for a delta fetch. When omitted, downloads full history
    (`period="max"`).

    Returns
    -------
    DataFrame with columns: snapped_at (UTC timestamp) & price (daily Close)
    """
    kwargs = dict(interval="1d", progress=False, auto_adjust=False)
    if start is None:
        df = yf.download(ticker, period="max", **kwargs)
    else:
        df = yf.download(ticker, start=start, **kwargs)

    if start is None and (df is None or df.empty):
        raise ValueError(f"No data returned for ticker '{ticker}'")

    return _normalize_downloaded(df)


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Load a raw price CSV and normalize types."""
    df = pd.read_csv(path)
    if "snapped_at" not in df.columns or "price" not in df.columns:
        raise ValueError(f"raw CSV at {path} is missing snapped_at / price columns")
    df["snapped_at"] = pd.to_datetime(df["snapped_at"], utc=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").round(8)
    df = df.dropna(subset=["snapped_at", "price"])
    return df[["snapped_at", "price"]].sort_values("snapped_at").reset_index(drop=True)


def merge_raw_prices(cached: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Concat, de-duplicate by UTC calendar day; later rows (the fetch) win."""
    parts = [_flatten_columns(d.copy()) for d in (cached, incoming) if d is not None and len(d)]
    if not parts:
        return pd.DataFrame(columns=["snapped_at", "price"])
    both = pd.concat(parts, ignore_index=True)
    both["snapped_at"] = pd.to_datetime(both["snapped_at"], utc=True)
    both["_day"] = both["snapped_at"].dt.normalize()
    both = both.drop_duplicates(subset=["_day"], keep="last").drop(columns=["_day"])
    both["price"] = both["price"].round(8)
    return both.sort_values("snapped_at").reset_index(drop=True)


def load_or_fetch_raw(ticker: str, path: Path, cache_data: bool = False, refresh: bool = False) -> pd.DataFrame:
    """Full download by default. With cache_data, delta-fetch from last_date - 5 days.

    If the fetch fails or is rate-limited and a cache file exists, return the cache
    with a warning. `--refresh` forces a full period=max download.
    """
    path = Path(path)
    cached = None
    if cache_data and path.exists() and not refresh:
        try:
            cached = read_raw_csv(path)
        except Exception as e:
            warnings.warn(f"Could not read cached raw data at {path} ({e}); doing a full download", RuntimeWarning)
            cached = None

    if cache_data and cached is not None and not refresh and len(cached):
        start = (cached["snapped_at"].max() - pd.Timedelta(days=DELTA_LOOKBACK_DAYS)).date()
        try:
            incoming = fetch_historical_data(ticker, start=start)
            return merge_raw_prices(cached, incoming)
        except (YFRateLimitError, YFException, OSError, ValueError) as e:
            warnings.warn(
                f"Yahoo fetch failed ({type(e).__name__}: {e}); using cached raw data from {path}",
                RuntimeWarning,
            )
            return cached

    try:
        return fetch_historical_data(ticker)
    except (YFRateLimitError, YFException, OSError, ValueError) as e:
        if cache_data:
            if cached is None and path.exists():
                try:
                    cached = read_raw_csv(path)
                except Exception:
                    cached = None
            if cached is not None and len(cached):
                warnings.warn(
                    f"Yahoo fetch failed ({type(e).__name__}: {e}); using cached raw data from {path}",
                    RuntimeWarning,
                )
                return cached
        raise


def save_raw_data(df: pd.DataFrame, output_path: Path):
    """
    Save DataFrame to CSV, ensuring directory exists.

    Parameters
    ----------
    df: Data to save
    output_path: Path to output CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[["snapped_at", "price"]].to_csv(output_path, index=False, header=["snapped_at", "price"])
    print(f"Raw data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch raw historical crypto data from Yahoo Finance")
    parser.add_argument("--coin_symbol", type=str, default="btc", help="Coin symbol for filename (e.g., 'btc')")
    parser.add_argument("--vs_currency", type=str, default="usd", help="Currency to price against (default: 'usd')")
    parser.add_argument("--cache_data", action="store_true", default=False,
                        help="If a raw CSV exists, re-fetch only from last date minus 5 days and merge")
    parser.add_argument("--refresh", action="store_true", default=False,
                        help="Force a full period=max download, even with --cache_data")

    args = parser.parse_args()

    ticker = f"{args.coin_symbol.upper()}-{args.vs_currency.upper()}"

    base_dir = Path(__file__).resolve().parents[1]
    output_csv = base_dir / "data" / "raw" / f"{args.coin_symbol}-{args.vs_currency}-max.csv"

    print(f"Fetching data for {ticker}...")

    df = load_or_fetch_raw(ticker, output_csv, cache_data=args.cache_data, refresh=args.refresh)
    save_raw_data(df, output_csv)
