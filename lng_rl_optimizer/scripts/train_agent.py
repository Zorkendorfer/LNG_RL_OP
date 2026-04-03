#!/usr/bin/env python
"""Train the PPO RL agent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import mlflow


@click.command()
@click.option("--total-steps",    default=5_000_000, show_default=True)
@click.option("--n-envs",         default=4, show_default=True)
@click.option("--lr",             default=3e-4, show_default=True)
@click.option("--batch-size",     default=2048, show_default=True)
@click.option("--output-dir",     default="runs/agent")
@click.option("--surrogate",      default="runs/surrogate/best_pinn.pt")
@click.option("--synthetic-prices", is_flag=True)
def train(total_steps, n_envs, lr, batch_size, output_dir, surrogate, synthetic_prices):
    """Train PPO agent on the LNG terminal environment."""
    from src.environment.lng_terminal_env import LNGTerminalEnv

    use_surrogate = Path(surrogate).exists()
    if not use_surrogate:
        print(f"Warning: surrogate not found at {surrogate}, using physics sim")

    def make_env():
        return LNGTerminalEnv(
            surrogate_path=surrogate,
            use_surrogate=use_surrogate,
            use_synthetic_prices=synthetic_prices,
            episode_length_h=8760,
        )

    vec_env  = make_vec_env(make_env, n_envs=n_envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    import torch.nn as nn
    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=f"{output_dir}/tb",
        policy_kwargs=dict(net_arch=[256, 256, 128], activation_fn=nn.Tanh),
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=f"{output_dir}/best",
            log_path=f"{output_dir}/eval_logs",
            eval_freq=50_000,
            n_eval_episodes=3,
            deterministic=True,
        ),
        CheckpointCallback(save_freq=200_000, save_path=f"{output_dir}/checkpoints"),
    ]

    with mlflow.start_run(run_name="ppo_lng_agent"):
        mlflow.log_params({
            "total_steps": total_steps, "n_envs": n_envs,
            "lr": lr, "batch_size": batch_size,
        })
        model.learn(total_timesteps=total_steps, callback=callbacks)
        model.save(f"{output_dir}/final_model")
        print(f"Model saved to {output_dir}/final_model")


if __name__ == "__main__":
    train()
