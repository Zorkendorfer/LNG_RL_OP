#!/usr/bin/env python
"""Generate synthetic NordPool-style LT hourly price CSVs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import pandas as pd

from src.market.nordpool_fetcher import generate_synthetic_prices


@click.command()
@click.option("--years", default="2020,2021,2022,2023,2024", show_default=True)
@click.option("--output-dir", default="data/nordpool/raw", show_default=True)
def main(years, output_dir):
    """Write yearly synthetic LT price CSVs compatible with the loader."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for year_str in years.split(","):
        year = int(year_str.strip())
        start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC")
        n_hours = int((end - start) / pd.Timedelta(hours=1))

        df = generate_synthetic_prices(n_hours=n_hours, seed=year)
        df.index = pd.date_range(start, periods=n_hours, freq="h", tz="UTC")
        export_df = df[["price_eur_mwh"]].copy()

        path = out / f"lt_prices_{year}.csv"
        export_df.to_csv(path)
        print(f"Saved {len(export_df):,} hourly prices -> {path}")


if __name__ == "__main__":
    main()
