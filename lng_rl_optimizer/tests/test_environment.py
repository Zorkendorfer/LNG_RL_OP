import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.environment.lng_terminal_env import LNGTerminalEnv


def make_env():
    return LNGTerminalEnv(
        use_surrogate=False,
        use_synthetic_prices=True,
        episode_length_h=24,
    )


def test_env_reset_shape():
    env = make_env()
    obs, info = env.reset()
    assert obs.shape == (44,)
    assert env.observation_space.contains(obs)


def test_env_step():
    env = make_env()
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs2, reward, done, trunc, info = env.step(action)
    assert obs2.shape == (44,)
    assert isinstance(reward, float)
    assert "cost_eur_h" in info
    assert not trunc


def test_env_done_after_episode():
    env = make_env()
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        steps += 1
    assert steps == 24, f"Expected 24 steps, got {steps}"


def test_env_price_in_obs():
    env = make_env()
    obs, _ = env.reset()
    price_norm = obs[7]
    assert 0.0 <= price_norm <= 2.0, f"Price norm {price_norm} out of range"
