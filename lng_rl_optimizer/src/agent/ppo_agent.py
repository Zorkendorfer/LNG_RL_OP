"""
PPO agent wrapper — thin layer over stable-baselines3 PPO.
Training is done via scripts/train_agent.py.
"""
from stable_baselines3 import PPO
from pathlib import Path


def load_agent(model_path: str) -> PPO:
    return PPO.load(model_path)


def make_ppo(env, lr: float = 3e-4, batch_size: int = 2048) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
        ),
    )
