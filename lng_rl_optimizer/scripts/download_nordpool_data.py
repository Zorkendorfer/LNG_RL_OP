#!/usr/bin/env python
"""Download NordPool Baltic electricity price data via ENTSO-E API."""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd


def fetch_entsoe_prices(
    start: str,
    end: str,
    bidding_zone: str = "10YLT-1001A0008Q",
    output_dir: Path = Path("data/nordpool/raw"),
) -> pd.DataFrame:
    token = os.environ.get("ENTSOE_TOKEN")
    if not token:
        print(
            "No ENTSOE_TOKEN found. Register free at transparency.entsoe.eu\n"
            "Manual download instructions:\n"
            "  1. Go to https://transparency.entsoe.eu/\n"
            "  2. Market Data > Day-Ahead Prices > LT (Lithuania)\n"
            "  3. Select date range and export as CSV\n"
            "  4. Save to data/nordpool/raw/lt_prices_YYYY.csv\n"
        )
        return pd.DataFrame()

    try:
        from entsoe import EntsoePandasClient
    except ImportError:
        print("Install entsoe-py: pip install entsoe-py")
        return pd.DataFrame()

    client = EntsoePandasClient(api_key=token)
    prices = client.query_day_ahead_prices(
        "LT",
        start=pd.Timestamp(start, tz="Europe/Vilnius"),
        end=pd.Timestamp(end, tz="Europe/Vilnius"),
    )
    return prices


@click.command()
@click.option("--years", default="2020,2021,2022,2023,2024")
@click.option("--output-dir", default="data/nordpool/raw")
def download(years, output_dir):
    """Download historical LT electricity prices from ENTSO-E."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for year in years.split(","):
        year = year.strip()
        print(f"Fetching {year}...")
        prices = fetch_entsoe_prices(f"{year}0101", f"{year}1231", output_dir=out)
        if not prices.empty:
            path = out / f"lt_prices_{year}.csv"
            prices.to_csv(path)
            print(f"  Saved {year}: {len(prices)} hourly prices → {path}")
        else:
            print(f"  No data for {year} (see instructions above)")


if __name__ == "__main__":
    download()
