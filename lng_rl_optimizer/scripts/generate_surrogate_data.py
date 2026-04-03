#!/usr/bin/env python
"""Generate physics simulator trajectories for PINN training."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import click
from src.surrogate.data_generator import generate_training_trajectories


@click.command()
@click.option("--n-episodes",    default=500, show_default=True)
@click.option("--episode-length", default=168, show_default=True, help="Hours per episode")
@click.option("--output-dir",    default="data/synthetic/terminal_trajectories")
@click.option("--seed",          default=42)
def main(n_episodes, episode_length, output_dir, seed):
    """Generate surrogate training data using the physics simulator."""
    config = yaml.safe_load(open("config/terminal.yaml"))
    out    = Path(output_dir)
    print(f"Generating {n_episodes} episodes of {episode_length}h each...")
    generate_training_trajectories(
        n_episodes=n_episodes,
        episode_length_h=episode_length,
        config=config,
        output_dir=out,
        seed=seed,
    )
    total_h = n_episodes * episode_length
    print(f"Done. {total_h:,} timesteps saved to {out}/")


if __name__ == "__main__":
    main()
