#!/usr/bin/env python
"""Train the 24-hour ahead electricity price forecaster."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from src.market.nordpool_fetcher import load_and_clean_prices, generate_synthetic_prices
from src.market.price_forecaster import train_forecaster


@click.command()
@click.option("--price-dir",  default="data/nordpool/raw")
@click.option("--output",     default="runs/price_forecaster.pt")
@click.option("--epochs",     default=50, show_default=True)
@click.option("--synthetic",  is_flag=True, help="Use synthetic prices if real data unavailable")
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    show_default=True,
)
def main(price_dir, output, epochs, synthetic, device):
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if synthetic:
        print("Using synthetic price data")
        prices_df = generate_synthetic_prices()
    else:
        prices_df = load_and_clean_prices(Path(price_dir))

    model = train_forecaster(
        prices_df, output_path=output, epochs=epochs, device=device
    )
    print(f"Price forecaster saved to {output}")


if __name__ == "__main__":
    main()
