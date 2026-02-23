"""
Author: Jake Vick
Purpose: ETL extract layer to fetch raw historical price data from CoinGecko API v3 and save as CSV.
This feeds into returns.py for log return computation.
"""

import requests
import pandas as pd
from pathlib import Path
import time
import argparse

def fetch_historical_data(coin_id: str, vs_currency: str = 'usd') -> pd.DataFrame:
    """
    Fetch historical market data for a cryptocurrency using CoinGecko API.

    Parameters
    ----------
    coin_id: CoinGecko coin ID (e.g., 'bitcoin')
    vs_currency: Currency to price against (default: 'usd')

    Returns
    -------
    DataFrame with columns: snapped_at, price, market_cap, total_volume
    """
    base_url = "https://api.coingecko.com/api/v3"
    endpoint = f"/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": "max",
        "interval": "daily"  # Ensures daily data for long ranges
    }

    response = requests.get(f"{base_url}{endpoint}", params=params)
    
    if response.status_code != 200:
        raise ValueError(f"API request failed with status {response.status_code}: {response.text}")
    
    data = response.json()
    
    if not all(key in data for key in ['prices', 'market_caps', 'total_volumes']):
        raise ValueError("Incomplete data returned from API")
    
    # Extract data into DataFrame
    df = pd.DataFrame({
        'snapped_at': [time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(ts / 1000)) for ts, _ in data['prices']],
        'price': [p for _, p in data['prices']],
        'market_cap': [m for _, m in data['market_caps']],
        'total_volume': [v for _, v in data['total_volumes']],
    })
    
    # Convert snapped_at to datetime for consistency
    df['snapped_at'] = pd.to_datetime(df['snapped_at'], utc=True)
    
    return df

def save_raw_data(df: pd.DataFrame, output_path: Path):
    """
    Save DataFrame to CSV, ensuring directory exists.

    Parameters
    ----------
    df: Data to save
    output_path: Path to output CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch raw historical crypto data from CoinGecko")
    parser.add_argument("--coin_id", type=str, default="bitcoin", help="CoinGecko coin ID (e.g., 'bitcoin')")
    parser.add_argument("--coin_symbol", type=str, default="btc", help="Coin symbol for filename (e.g., 'btc')")
    parser.add_argument("--vs_currency", type=str, default="usd", help="Currency to price against (default: 'usd')")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).resolve().parents[1]
    output_csv = base_dir / "data" / "raw" / f"{args.coin_symbol}-{args.vs_currency}-max.csv"
    
    df = fetch_historical_data(args.coin_id, args.vs_currency)
    save_raw_data(df, output_csv)