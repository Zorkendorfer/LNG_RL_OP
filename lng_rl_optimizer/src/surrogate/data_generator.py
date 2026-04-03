import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import yaml

from src.physics.terminal_simulator import LNGTerminalSimulator, TerminalState


def generate_training_trajectories(
    n_episodes: int,
    episode_length_h: int,
    config: dict,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """
    Generate diverse trajectories by running the physics simulator
    with random control actions and random initial conditions.
    Saves each episode as a parquet file.
    """
    rng = np.random.default_rng(seed)
    sim = LNGTerminalSimulator(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode in tqdm(range(n_episodes), desc="Generating trajectories"):
        x_me = rng.uniform(0.82, 0.95)
        x_et = rng.uniform(0.03, 0.12)
        x_pr = rng.uniform(0.01, 0.05)
        x_ni = 0.01
        tot  = x_me + x_et + x_pr + x_ni
        state = TerminalState(
            liquid_volume_m3=float(rng.uniform(30000, 155000)),
            tank_pressure_kPa=float(rng.uniform(3, 20)),
            lng_temp_K=float(rng.uniform(110, 115)),
            x_methane=x_me / tot,
            x_ethane=x_et / tot,
            x_propane=x_pr / tot,
            x_nitrogen=x_ni / tot,
            ambient_temp_C=float(rng.uniform(-10, 30)),
            seawater_temp_C=float(rng.uniform(2, 22)),
            sea_state=float(rng.uniform(0, 1)),
            electricity_price_eur_mwh=float(rng.uniform(0, 300)),
        )

        records = []
        for h in range(episode_length_h):
            action = {
                "compressor_loads": [
                    float(rng.choice([0.0, rng.uniform(0.3, 1.0)])),
                    float(rng.choice([0.0, rng.uniform(0.3, 1.0)])),
                    float(rng.choice([0.0, 0.0, 0.0, rng.uniform(0.3, 0.8)])),
                ],
                "n_hp_pumps":       int(rng.choice([1, 2, 3, 4])),
                "n_vaporizers":     int(rng.choice([1, 2, 3, 4])),
                "send_out_rate_t_h": float(rng.uniform(20, 380)),
            }

            new_state, info = sim.step(state, action)

            record = {
                "episode": episode, "hour": h,
                "fill_fraction":    state.fill_fraction,
                "tank_pressure_kPa": state.tank_pressure_kPa,
                "x_methane":        state.x_methane,
                "x_ethane":         state.x_ethane,
                "ambient_temp_C":   state.ambient_temp_C,
                "seawater_temp_C":  state.seawater_temp_C,
                "sea_state":        state.sea_state,
                "price_eur_mwh":    state.electricity_price_eur_mwh,
                "comp_load_0":      action["compressor_loads"][0],
                "comp_load_1":      action["compressor_loads"][1],
                "comp_load_2":      action["compressor_loads"][2],
                "n_hp_pumps":       action["n_hp_pumps"],
                "n_vaporizers":     action["n_vaporizers"],
                "send_out_rate":    action["send_out_rate_t_h"],
                "bog_gen_kg_h":     info["bog_gen_kg_h"],
                "comp_power_kW":    info["comp_power_kW"],
                "total_power_kW":   info["total_power_kW"],
                "cost_eur_h":       info["cost_eur_h"],
                "new_pressure_kPa": new_state.tank_pressure_kPa,
                "new_fill_fraction": new_state.fill_fraction,
            }
            records.append(record)
            state = new_state
            # Random price walk
            state.electricity_price_eur_mwh = float(
                np.clip(state.electricity_price_eur_mwh + rng.normal(0, 10), 0, 400)
            )

        df = pd.DataFrame(records)
        df.to_parquet(output_dir / f"episode_{episode:05d}.parquet")
