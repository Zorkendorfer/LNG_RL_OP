#!/usr/bin/env python
"""Quick environment validation — run this first after setup."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    print("Checking imports...")
    import numpy; print(f"  numpy {numpy.__version__}")
    import pandas; print(f"  pandas {pandas.__version__}")
    import torch; print(f"  torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    import gymnasium; print(f"  gymnasium {gymnasium.__version__}")
    import yaml; print("  pyyaml OK")
    import mlflow; print(f"  mlflow {mlflow.__version__}")
    try:
        import CoolProp; print(f"  CoolProp OK")
    except ImportError:
        print("  CoolProp NOT installed — pip install CoolProp")
    try:
        import stable_baselines3; print(f"  stable-baselines3 {stable_baselines3.__version__}")
    except ImportError:
        print("  stable-baselines3 NOT installed")
    print("Imports OK\n")


def check_physics():
    print("Checking physics model...")
    import yaml
    from src.physics.terminal_simulator import LNGTerminalSimulator, TerminalState

    config = yaml.safe_load(open("config/terminal.yaml"))
    sim    = LNGTerminalSimulator(config)
    state  = TerminalState(
        liquid_volume_m3=100000,
        tank_pressure_kPa=10,
        lng_temp_K=112,
    )
    action = {
        "compressor_loads":  [0.8, 0.6, 0.0],
        "n_hp_pumps":        2,
        "n_vaporizers":      2,
        "send_out_rate_t_h": 150,
    }
    new_state, info = sim.step(state, action)

    print(f"  BOG generation:  {info['bog_gen_kg_h']:.1f} kg/h (expect 100–400)")
    print(f"  Total power:     {info['total_power_kW']:.0f} kW (expect 500–5000)")
    print(f"  Electricity cost: €{info['cost_eur_h']:.2f}/h")
    print(f"  Tank pressure:   {info['tank_pressure']:.2f} kPa")
    print(f"  Fill fraction:   {info['fill_fraction']:.3f}")

    assert 100 < info["bog_gen_kg_h"] < 400, f"BOG rate unphysical: {info['bog_gen_kg_h']}"
    assert 500 < info["total_power_kW"] < 5000, f"Power unphysical: {info['total_power_kW']}"
    print("Physics model OK\n")


def check_environment():
    print("Checking Gymnasium environment...")
    from src.environment.lng_terminal_env import LNGTerminalEnv

    env = LNGTerminalEnv(use_surrogate=False, use_synthetic_prices=True, episode_length_h=24)
    obs, _ = env.reset()
    print(f"  Observation shape: {obs.shape} (expect (44,))")
    assert obs.shape == (44,), f"Wrong obs shape: {obs.shape}"
    assert env.observation_space.contains(obs), "Obs outside observation space"

    action = env.action_space.sample()
    obs2, reward, done, _, info = env.step(action)
    print(f"  Step reward: {reward:.2f}")
    print(f"  Cost EUR/h:  {info['cost_eur_h']:.2f}")
    assert obs2.shape == (44,)
    assert isinstance(reward, float)
    print("Environment OK\n")


if __name__ == "__main__":
    check_imports()
    check_physics()
    check_environment()
    print("All checks passed!")
