"""
Author: Jake Vick
Purpose: Compute log returns for Student-t VaR / CVaR modeling
"""

from pathlib import Path

import argparse
import pandas as pd
import numpy as np

def print_return_summary(r: pd.Series):
    """Print summary stats for return series."""
    print("\nReturn Summary Statistics")
    print("-" * 30)
    print(r.describe()[["min", "25%", "50%", "mean", "75%", "max"]])
    print(f"\nStd Dev: {r.std():.6f}")
    print(f"Excess Kurtosis: {r.kurt():.2f}")

def compute_log_returns(
    input_path: Path,
    output_path: Path,
    price_col: str = "price",
    date_col: str = "snapped_at",
    verbose: bool = True,
):
    """
    Read raw price data, compute log returns, and save to output. 

    Parameters ----------
    input_path: Path to raw CSV file
    output_path: Path to output CSV file
    price_col: Column containing price data
    date_col: Column containing timestamp

    Output ----------
    output/data.csv
    """

    # Load raw data
    df = pd.read_csv(input_path)

    # Parse datetime
    df[date_col] = pd.to_datetime(df[date_col], utc=True)

    # Sort chronologically 
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=[date_col]) # simple check for dups just to clean data if needed

    # Compute log returns
    prices = df[price_col].astype(float) # sometimes prices come in as strings
    df["return"] = np.log(prices / prices.shift(1))

    # Drop first NaN return
    df = df.dropna(subset=["return"])

    # Keep date (YYYY-MM-DD)
    df["date"] = df[date_col].dt.date

    # Summary Stats
    if verbose:
        print_return_summary(df["return"])

    # Select final columns
    out = df[["date", "return"]]

    # Reverse order: most recent first
    out = out.sort_values("date", ascending=False)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    out.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute log returns from raw crypto price data")
    parser.add_argument(
        "--coin_symbol",
        type=str,
        default="btc",
        help="Coin symbol (e.g., 'btc', 'eth', 'sol') — matches api.py filename"
    )
    parser.add_argument(
        "--vs_currency",
        type=str,
        default="usd",
        help="Quote currency (default: usd)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print summary statistics (default: True)"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]

    input_csv = base_dir / "data" / "raw" / f"{args.coin_symbol}-{args.vs_currency}-max.csv"
    output_csv = base_dir / "data" / "output" / f"{args.coin_symbol}-log-returns.csv"

    if not input_csv.exists():
        print(f"Error: Raw data file not found: {input_csv}")
        print("Run api.py first: python api.py --coin_symbol", args.coin_symbol)
        exit(1)

    compute_log_returns(input_csv, output_csv, verbose=args.verbose)