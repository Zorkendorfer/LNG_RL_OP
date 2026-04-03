import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.physics.terminal_simulator import TerminalState
from src.environment.reward import compute_reward


def get_config():
    return yaml.safe_load(open("config/terminal.yaml"))


def make_state(**kwargs):
    defaults = dict(
        liquid_volume_m3=100000,
        tank_pressure_kPa=10.0,
        lng_temp_K=112.0,
        electricity_price_eur_mwh=50.0,
    )
    defaults.update(kwargs)
    return TerminalState(**defaults)


PENALTY_WEIGHTS = {
    "pressure": 100.0, "low_inventory": 500.0,
    "high_inventory": 100.0, "bog_flaring": 50.0, "comp_start": 500.0,
}


def test_higher_cost_lower_reward():
    config = get_config()
    state  = make_state()
    info_cheap = {"cost_eur_h": 50.0,  "total_power_kW": 1000, "bog_gen_kg_h": 200, "bog_removed_kg_h": 200}
    info_expensive = {"cost_eur_h": 200.0, "total_power_kW": 1000, "bog_gen_kg_h": 200, "bog_removed_kg_h": 200}
    r_cheap, _    = compute_reward(state, state, info_cheap,    config, PENALTY_WEIGHTS)
    r_expensive, _ = compute_reward(state, state, info_expensive, config, PENALTY_WEIGHTS)
    assert r_cheap > r_expensive


def test_overpressure_penalty():
    config     = get_config()
    state      = make_state()
    high_p_state = make_state(tank_pressure_kPa=24.0)  # above max_pressure_kPa=22
    info = {"cost_eur_h": 100, "total_power_kW": 2000, "bog_gen_kg_h": 200, "bog_removed_kg_h": 200}
    r_normal, _ = compute_reward(state, state,      info, config, PENALTY_WEIGHTS)
    r_high_p, _ = compute_reward(state, high_p_state, info, config, PENALTY_WEIGHTS)
    assert r_high_p < r_normal, "Overpressure should decrease reward"
