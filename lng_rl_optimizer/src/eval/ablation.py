"""
Ablation study: remove each novelty and measure impact on annual cost.

Ablations:
A) No price forecast (obs[8:32] zeroed) — value of price forecasting
B) No composition tracking (x_methane, x_ethane fixed) — value of composition state
C) No FSRU motion (sea_state always 0) — value of FSRU-specific modeling
D) No carbon cost in reward — carbon cost impact on behavior
E) Full model (all novelties) — best performance

Present as a table: Ablation | Annual Cost | Savings vs Baseline | Delta vs Full
"""
import numpy as np
from copy import deepcopy


class AblationWrapper:
    """Wraps the environment to remove specific features for ablation."""

    def __init__(self, env, ablation: str):
        self.env      = env
        self.ablation = ablation
        self.observation_space = env.observation_space
        self.action_space      = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._modify_obs(obs), info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        obs = self._modify_obs(obs)

        if self.ablation == "no_carbon":
            reward += info.get("carbon_cost", 0)  # remove carbon penalty

        return obs, reward, done, trunc, info

    def _modify_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.copy()
        if self.ablation == "no_price_forecast":
            obs[8:32] = 0.0
        elif self.ablation == "no_composition":
            obs[2] = 0.90  # fixed nominal methane
            obs[3] = 0.07  # fixed nominal ethane
        elif self.ablation == "no_sea_state":
            obs[6] = 0.0
        return obs
