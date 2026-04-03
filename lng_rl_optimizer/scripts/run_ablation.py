#!/usr/bin/env python
"""Run ablation study: remove each novelty and measure cost impact."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from src.environment.lng_terminal_env import LNGTerminalEnv
from src.eval.ablation import AblationWrapper
from src.eval.metrics import evaluate_agent


ABLATIONS = {
    "full":             None,
    "no_price_forecast": "no_price_forecast",
    "no_composition":   "no_composition",
    "no_sea_state":     "no_sea_state",
    "no_carbon":        "no_carbon",
}


@click.command()
@click.option("--model-path",    default="runs/agent/best/best_model")
@click.option("--n-episodes",    default=3, show_default=True)
@click.option("--surrogate",     default="runs/surrogate/best_pinn.pt")
@click.option("--synthetic-prices", is_flag=True)
def ablation(model_path, n_episodes, surrogate, synthetic_prices):
    """Run ablation study showing contribution of each novelty."""
    use_surrogate = Path(surrogate).exists()
    agent = PPO.load(model_path)
    results = []

    for name, ablation_type in ABLATIONS.items():
        env = LNGTerminalEnv(
            surrogate_path=surrogate,
            use_surrogate=use_surrogate,
            use_synthetic_prices=synthetic_prices,
            episode_length_h=8760,
        )
        if ablation_type:
            env = AblationWrapper(env, ablation_type)

        print(f"\nAblation: {name}")
        metrics = evaluate_agent(env, agent, n_episodes)
        results.append({
            "ablation": name,
            "mean_cost": metrics["mean_annual_cost"],
            "price_corr": metrics["price_correlation"],
        })

    df = pd.DataFrame(results).set_index("ablation")
    full_cost = df.loc["full", "mean_cost"]
    df["delta_vs_full"] = df["mean_cost"] - full_cost

    print("\n=== Ablation Results ===")
    print(df.to_string())


if __name__ == "__main__":
    ablation()
