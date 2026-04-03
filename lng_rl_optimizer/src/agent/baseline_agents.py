import numpy as np


class RuleBasedAgent:
    """
    Current industry practice: fixed compressor count based on tank pressure.
    No price awareness, no forecasting.
    This is what the RL agent competes against.
    """

    def predict(self, obs: np.ndarray, state=None, episode_start=None) -> tuple:
        pressure = obs[1] * 25.0
        if pressure > 18:
            action = [3, 3, 0, 2, 2]
        elif pressure > 12:
            action = [2, 0, 0, 2, 2]
        else:
            action = [1, 0, 0, 1, 1]
        return np.array(action), state


class PriceAwareHeuristic:
    """
    Improved baseline: pressure-based logic that shifts loads to cheap hours.
    Demonstrates value of price awareness without RL complexity.
    """

    def predict(self, obs: np.ndarray, state=None, episode_start=None) -> tuple:
        pressure   = obs[1] * 25.0
        price      = obs[7] * 300.0
        future_min = obs[8:32].min() * 300.0
        price_high = price > future_min * 1.5

        if pressure > 20:
            action = [3, 3, 0, 2, 2]
        elif pressure > 15 and not price_high:
            action = [2, 2, 0, 2, 2]
        elif pressure > 10 and not price_high:
            action = [2, 0, 0, 2, 2]
        elif price_high:
            action = [1, 0, 0, 1, 2]
        else:
            action = [1, 0, 0, 1, 1]

        return np.array(action), state
