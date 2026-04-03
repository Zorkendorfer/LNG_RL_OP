#!/usr/bin/env python
"""Evaluate trained RL agent and baselines. Computes cost savings."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
from stable_baselines3 import PPO
from src.environment.lng_terminal_env import LNGTerminalEnv
from src.agent.baseline_agents import RuleBasedAgent, PriceAwareHeuristic
from src.eval.metrics import annual_cost_savings, evaluate_agent


@click.command()
@click.option("--model-path",    default="runs/agent/best/best_model")
@click.option("--n-episodes",    default=5, show_default=True)
@click.option("--synthetic-prices", is_flag=True)
@click.option("--surrogate",     default="runs/surrogate/best_pinn.pt")
def evaluate(model_path, n_episodes, synthetic_prices, surrogate):
    """Evaluate RL agent vs baselines and report cost savings."""
    use_surrogate = Path(surrogate).exists()

    def make_env(seed=42):
        return LNGTerminalEnv(
            surrogate_path=surrogate,
            use_surrogate=use_surrogate,
            use_synthetic_prices=synthetic_prices,
            episode_length_h=8760,
            seed=seed,
        )

    env = make_env()

    # RL agent
    print("=== RL Agent ===")
    rl_agent = PPO.load(model_path)
    rl_metrics = evaluate_agent(env, rl_agent, n_episodes)
    print(f"Mean annual cost: €{rl_metrics['mean_annual_cost']:,.0f}")
    print(f"Price correlation: {rl_metrics['price_correlation']:.3f}")

    # Rule-based baseline
    print("\n=== Rule-Based Baseline ===")
    rb_agent   = RuleBasedAgent()
    rb_metrics = evaluate_agent(make_env(seed=100), rb_agent, n_episodes)
    print(f"Mean annual cost: €{rb_metrics['mean_annual_cost']:,.0f}")

    # Price-aware heuristic
    print("\n=== Price-Aware Heuristic ===")
    pa_agent   = PriceAwareHeuristic()
    pa_metrics = evaluate_agent(make_env(seed=200), pa_agent, n_episodes)
    print(f"Mean annual cost: €{pa_metrics['mean_annual_cost']:,.0f}")

    # Summary
    print("\n=== Cost Savings Summary ===")
    savings_vs_rb = annual_cost_savings(
        [rl_metrics["mean_annual_cost"]], [rb_metrics["mean_annual_cost"]]
    )
    print(f"RL vs Rule-Based: €{savings_vs_rb['savings_eur']:,.0f} "
          f"({savings_vs_rb['savings_pct']:.1f}% savings)")

    savings_vs_pa = annual_cost_savings(
        [rl_metrics["mean_annual_cost"]], [pa_metrics["mean_annual_cost"]]
    )
    print(f"RL vs Price-Aware: €{savings_vs_pa['savings_eur']:,.0f} "
          f"({savings_vs_pa['savings_pct']:.1f}% savings)")


if __name__ == "__main__":
    evaluate()
