import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_episode(
    prices: np.ndarray,
    actions: np.ndarray,
    costs: np.ndarray,
    pressures: np.ndarray,
    output_path: Path = None,
):
    """Plot price, compressor load, cost, and pressure over an episode."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(prices, color="tab:blue", alpha=0.8)
    axes[0].set_ylabel("Price (EUR/MWh)")
    axes[0].set_title("LNG Terminal RL Agent — Episode Summary")

    comp_load = actions[:, 0] + actions[:, 1] + actions[:, 2] * 0.5
    axes[1].plot(comp_load, color="tab:orange")
    axes[1].set_ylabel("Compressor Load (norm)")

    axes[2].plot(costs, color="tab:green")
    axes[2].set_ylabel("Cost (EUR/h)")

    axes[3].plot(pressures, color="tab:red")
    axes[3].axhline(22.0, color="red",   linestyle="--", alpha=0.5, label="Max")
    axes[3].axhline(2.0,  color="blue",  linestyle="--", alpha=0.5, label="Min")
    axes[3].set_ylabel("Tank Pressure (kPa)")
    axes[3].set_xlabel("Hour")
    axes[3].legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    return fig


def plot_price_load_scatter(
    prices: np.ndarray,
    comp_loads: np.ndarray,
    title: str = "Price vs Compressor Load",
    output_path: Path = None,
):
    """Scatter plot showing price-load relationship (novelty verification)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(prices, comp_loads, alpha=0.1, s=2)
    ax.set_xlabel("Electricity Price (EUR/MWh)")
    ax.set_ylabel("Compressor Load (normalized)")
    ax.set_title(title)

    corr = np.corrcoef(prices, comp_loads)[0, 1]
    ax.text(0.05, 0.95, f"Correlation: {corr:.3f}",
            transform=ax.transAxes, fontsize=12,
            verticalalignment="top")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    return fig
