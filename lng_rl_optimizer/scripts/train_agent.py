#!/usr/bin/env python
"""Train the PPO RL agent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import mlflow
from src.utils.device import resolve_torch_device
from tqdm.auto import tqdm


class TqdmStepCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self._bar = None

    def _on_training_start(self) -> None:
        self._bar = tqdm(
            total=self.total_timesteps,
            desc="PPO training",
            unit="step",
        )

    def _on_step(self) -> bool:
        if self._bar is not None:
            delta = self.model.num_timesteps - self._bar.n
            if delta > 0:
                self._bar.update(delta)
                if len(self.model.ep_info_buffer) > 0:
                    last = self.model.ep_info_buffer[-1]
                    reward = last.get("r")
                    length = last.get("l")
                    if reward is not None and length is not None:
                        self._bar.set_postfix(
                            ep_reward=f"{reward:.2f}",
                            ep_len=int(length),
                        )
        return True

    def _on_training_end(self) -> None:
        if self._bar is not None:
            self._bar.close()


@click.command()
@click.option("--total-steps",    default=5_000_000, show_default=True)
@click.option("--n-envs",         default=4, show_default=True)
@click.option("--lr",             default=3e-4, show_default=True)
@click.option("--batch-size",     default=2048, show_default=True)
@click.option("--output-dir",     default="runs/agent")
@click.option("--surrogate",      default="runs/surrogate/best_pinn.pt")
@click.option("--synthetic-prices", is_flag=True)
@click.option(
    "--vec-env",
    default="auto",
    type=click.Choice(["auto", "dummy", "subproc"]),
    show_default=True,
)
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    show_default=True,
)
@click.option(
    "--env-device",
    default="auto",
    type=click.Choice(["auto", "cuda", "mps", "cpu"]),
    show_default=True,
)
def train(total_steps, n_envs, lr, batch_size, output_dir, surrogate, synthetic_prices, vec_env, device, env_device):
    """Train PPO agent on the LNG terminal environment."""
    from src.environment.lng_terminal_env import LNGTerminalEnv

    use_surrogate = Path(surrogate).exists()
    if not use_surrogate:
        print(f"Warning: surrogate not found at {surrogate}, using physics sim")
    resolved_device = resolve_torch_device(device)
    if env_device == "auto":
        resolved_env_device = "cpu" if use_surrogate else "cpu"
    else:
        resolved_env_device = resolve_torch_device(env_device)
    if vec_env == "auto":
        if use_surrogate and resolved_env_device == "cpu" and n_envs > 1:
            vec_env_cls = SubprocVecEnv
            vec_env_name = "subproc"
        elif resolved_env_device in {"cuda", "mps"}:
            vec_env_cls = DummyVecEnv
            vec_env_name = "dummy"
        else:
            vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
            vec_env_name = "subproc" if n_envs > 1 else "dummy"
    elif vec_env == "dummy":
        vec_env_cls = DummyVecEnv
        vec_env_name = "dummy"
    else:
        vec_env_cls = SubprocVecEnv
        vec_env_name = "subproc"

    eval_vec_env_cls = vec_env_cls if vec_env_cls is not SubprocVecEnv else SubprocVecEnv
    print(f"Using torch device: {resolved_device}")
    print(
        f"Training PPO for {total_steps:,} steps with {n_envs} envs, "
        f"batch_size={batch_size}, lr={lr}"
    )
    print(
        f"Price source: {'synthetic' if synthetic_prices else 'data/nordpool/raw'} | "
        f"Surrogate: {'enabled' if use_surrogate else 'physics simulator'} | "
        f"Vec env: {vec_env_name} | Env device: {resolved_env_device}"
    )

    def make_env():
        return LNGTerminalEnv(
            surrogate_path=surrogate,
            use_surrogate=use_surrogate,
            use_synthetic_prices=synthetic_prices,
            episode_length_h=8760,
            surrogate_device=resolved_env_device,
            fast_training=True,
        )

    vec_env  = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=vec_env_cls)
    eval_env = make_vec_env(make_env, n_envs=1, vec_env_cls=eval_vec_env_cls)

    import torch.nn as nn
    model = PPO(
        "MlpPolicy", vec_env,
        device=resolved_device,
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
        TqdmStepCallback(total_steps),
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
            "lr": lr, "batch_size": batch_size, "device": resolved_device,
            "vec_env": vec_env_name, "env_device": resolved_env_device,
        })
        model.learn(total_timesteps=total_steps, callback=callbacks)
        model.save(f"{output_dir}/final_model")
        print(f"Model saved to {output_dir}/final_model")


if __name__ == "__main__":
    train()
