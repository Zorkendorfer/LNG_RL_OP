"""
Safe PPO placeholder — extends standard PPO with constraint projection.
For the initial implementation, use the SafetyLayer in the environment.
Full CBF-based safe RL can be added as a future extension.
"""
from .ppo_agent import make_ppo


def make_safe_ppo(env, **kwargs):
    """
    Creates PPO agent with safety handled by the environment's safety layer.
    The environment projects unsafe actions before executing them.
    """
    return make_ppo(env, **kwargs)
