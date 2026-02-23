"""
Author: Jake Vick
Purpose: ETL extract layer to fetch raw historical price data from yfinance and save as CSV.
This feeds into returns.py for log return computation.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
import argparse

def fetch_historical_data(ticker: str) -> pd.DataFrame:
    """
    Fetch historical market data for a cryptocurrency using yfinance.

    Parameters
    ----------
    ticker: Yahoo Finance ticker symbol (e.g., 'BTC-USD', 'ETH-USD', 'SOL-USD')
    
    Returns
    -------
    DataFrame with columns: snapped_at & close_price
    """
    # Download daily data, full history available
    df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")

    # Keep only Date and Close, rename to match returns.py expectations
    df = df[['Close']].reset_index()
    df = df.rename(columns={
        'Date': 'snapped_at',
        'Close': 'price'
    })

    # Ensure datetime is timezone-aware (UTC)
    df['snapped_at'] = pd.to_datetime(df['snapped_at'], utc=True)

    df['price'] = df['price'].round(8)

    return df[['snapped_at', 'price']]

def save_raw_data(df: pd.DataFrame, output_path: Path):
    """
    Save DataFrame to CSV, ensuring directory exists.

    Parameters
    ----------
    df: Data to save
    output_path: Path to output CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[['snapped_at', 'price']].to_csv(output_path, index=False, header=['snapped_at', 'price'])
    print(f"Raw data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch raw historical crypto data from Yahoo Finance")
    parser.add_argument("--coin_symbol", type=str, default="btc", help="Coin symbol for filename (e.g., 'btc')")
    parser.add_argument("--vs_currency", type=str, default="usd", help="Currency to price against (default: 'usd')")
    
    args = parser.parse_args()

    # Construct Yahoo ticker (ex BTC-USD)
    ticker = f"{args.coin_symbol.upper()}-{args.vs_currency.upper()}"
    
    base_dir = Path(__file__).resolve().parents[1]
    output_csv = base_dir / "data" / "raw" / f"{args.coin_symbol}-{args.vs_currency}-max.csv"

    print(f"Fetching data for {ticker}...")
    
    df = fetch_historical_data(ticker)
    save_raw_data(df, output_csv)