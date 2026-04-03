#!/usr/bin/env python
"""Train the physics-informed neural network surrogate."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from src.surrogate.trainer import train_surrogate


@click.command()
@click.option("--data-dir",      default="data/synthetic/terminal_trajectories")
@click.option("--output-dir",    default="runs/surrogate")
@click.option("--hidden-dim",    default=256, show_default=True)
@click.option("--n-layers",      default=6,   show_default=True)
@click.option("--epochs",        default=200, show_default=True)
@click.option("--batch-size",    default=1024, show_default=True)
@click.option("--lr",            default=3e-4, show_default=True)
@click.option("--physics-weight", default=0.5, show_default=True)
def main(data_dir, output_dir, hidden_dim, n_layers, epochs, batch_size, lr, physics_weight):
    """Train PINN surrogate on physics simulator trajectories."""
    model = train_surrogate(
        data_dir=Path(data_dir),
        output_dir=Path(output_dir),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        physics_weight=physics_weight,
    )
    print(f"Surrogate saved to {output_dir}/best_pinn.pt")


if __name__ == "__main__":
    main()
