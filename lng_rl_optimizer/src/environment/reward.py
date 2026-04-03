import numpy as np


def compute_reward(
    state,
    new_state,
    info: dict,
    config: dict,
    penalty_weights: dict,
) -> tuple:
    """
    Reward = -electricity_cost - constraint_penalties - wear_costs - carbon_cost
    Returns: (reward, info_dict with component breakdown)
    """
    cost_eur_h = info["cost_eur_h"]
    reward = -cost_eur_h
    penalties = {}

    pressure = new_state.tank_pressure_kPa
    max_p    = config["tanks"]["max_pressure_kPa"]
    min_p    = config["tanks"]["min_pressure_kPa"]
    if pressure > max_p:
        pen = penalty_weights["pressure"] * (pressure - max_p) ** 2
        reward -= pen
        penalties["overpressure"] = pen
    elif pressure < min_p:
        pen = penalty_weights["pressure"] * (min_p - pressure) ** 2
        reward -= pen
        penalties["underpressure"] = pen

    fill = new_state.fill_fraction
    if fill < 0.12:
        pen = penalty_weights["low_inventory"] * (0.12 - fill) * 1000
        reward -= pen
        penalties["low_inventory"] = pen
    elif fill > 0.98:
        pen = penalty_weights["high_inventory"] * (fill - 0.98) * 1000
        reward -= pen
        penalties["high_inventory"] = pen

    bog_gen     = info["bog_gen_kg_h"]
    bog_removed = info["bog_removed_kg_h"]
    bog_excess  = max(0, bog_gen - bog_removed)
    if bog_excess > 50:
        pen = penalty_weights["bog_flaring"] * bog_excess
        reward -= pen
        penalties["bog_flaring"] = pen

    n_new_starts = (
        new_state.compressor_starts_today - state.compressor_starts_today
    )
    if n_new_starts > 0:
        pen = penalty_weights["comp_start"] * n_new_starts
        reward -= pen
        penalties["compressor_wear"] = pen

    grid_emission_factor = 0.30
    carbon_price_eur_t   = 65.0
    energy_mwh  = info["total_power_kW"] / 1000
    carbon_cost = energy_mwh * grid_emission_factor * carbon_price_eur_t
    reward -= carbon_cost
    penalties["carbon_cost"] = carbon_cost

    info_out = {
        "reward_total":     reward,
        "electricity_cost": cost_eur_h,
        "carbon_cost":      carbon_cost,
        **penalties,
    }

    return float(reward), info_out
