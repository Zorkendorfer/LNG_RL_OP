import pandas as pd
import numpy as np
from pathlib import Path


def load_and_clean_prices(data_dir: Path) -> pd.DataFrame:
    """
    Load all yearly CSVs, concatenate, clean, and return hourly price series.
    Returns DataFrame with columns: [timestamp, price_eur_mwh, day_of_week,
                                     hour, month, is_weekend, price_log]
    """
    dfs = []
    for csv_path in sorted(data_dir.glob("lt_prices_*.csv")):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No price CSVs found in {data_dir}. "
            "Run scripts/download_nordpool_data.py first."
        )

    prices = pd.concat(dfs).sort_index()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_convert("UTC")

    prices = prices[~prices.index.duplicated(keep="first")]

    n_missing = prices.isna().sum().sum()
    if n_missing > 0:
        print(f"Filling {n_missing} missing hours")
        prices = prices.ffill(limit=4)

    prices = prices.rename(columns={prices.columns[0]: "price_eur_mwh"})
    prices["hour"]        = prices.index.hour
    prices["day_of_week"] = prices.index.dayofweek
    prices["month"]       = prices.index.month
    prices["is_weekend"]  = prices.index.dayofweek >= 5
    prices["price_log"]   = np.log1p(prices["price_eur_mwh"].clip(lower=0))

    print(f"Price range: €{prices.price_eur_mwh.min():.1f} – "
          f"€{prices.price_eur_mwh.max():.1f}/MWh")
    print(f"Mean: €{prices.price_eur_mwh.mean():.1f}/MWh, "
          f"Std: €{prices.price_eur_mwh.std():.1f}/MWh")

    return prices


def generate_synthetic_prices(
    n_hours: int = 87600,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic electricity price series with realistic patterns
    for use when real NordPool data is unavailable.
    Includes: daily seasonality, weekly pattern, price spikes.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(n_hours)

    # Daily pattern (peak morning + evening)
    daily = (
        30 * np.sin(2 * np.pi * (hours % 24 - 6) / 24) +
        20 * np.sin(4 * np.pi * (hours % 24 - 4) / 24)
    )
    # Weekly pattern (lower on weekends)
    weekly = -10 * ((hours // 24) % 7 >= 5).astype(float)
    # Seasonal pattern (higher in winter)
    seasonal = 20 * np.cos(2 * np.pi * hours / 8760)
    # Base price + noise
    base   = 60.0
    noise  = rng.normal(0, 8, n_hours)
    # Occasional price spikes
    spikes = np.zeros(n_hours)
    spike_idx = rng.choice(n_hours, size=n_hours // 200, replace=False)
    spikes[spike_idx] = rng.exponential(150, len(spike_idx))

    price = np.clip(base + daily + weekly + seasonal + noise + spikes, -10, 500)

    idx = pd.date_range("2020-01-01", periods=n_hours, freq="h", tz="UTC")
    df  = pd.DataFrame({"price_eur_mwh": price}, index=idx)
    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["is_weekend"]  = df.index.dayofweek >= 5
    df["price_log"]   = np.log1p(df["price_eur_mwh"].clip(lower=0))
    return df
