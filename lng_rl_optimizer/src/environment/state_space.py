import numpy as np

# Full state vector (44 dimensions)
STATE_DIM = 44


def encode_state(state, price_forecast_24h: np.ndarray, hour: int = 0, day_of_week: int = 0) -> np.ndarray:
    obs = np.zeros(STATE_DIM, dtype=np.float32)
    obs[0]  = state.fill_fraction
    obs[1]  = state.tank_pressure_kPa / 25.0
    obs[2]  = state.x_methane
    obs[3]  = state.x_ethane
    obs[4]  = (state.ambient_temp_C + 20) / 55.0
    obs[5]  = (state.seawater_temp_C - 2) / 20.0
    obs[6]  = state.sea_state
    obs[7]  = state.electricity_price_eur_mwh / 300.0
    obs[8:32] = np.clip(price_forecast_24h / 300.0, 0, 1)
    obs[32] = np.sin(2 * np.pi * hour / 24)
    obs[33] = np.cos(2 * np.pi * hour / 24)
    obs[34] = day_of_week / 6.0
    obs[35] = state.compressor_loads[0]
    obs[36] = state.compressor_loads[1]
    obs[37] = state.compressor_loads[2]
    obs[38] = state.n_hp_pumps_running / 4.0
    obs[39] = state.n_vaporizers_running / 4.0
    obs[40] = min(state.compressor_starts_today / 10.0, 1.0)
    obs[41] = min(state.runtime_hours[0] / 8760.0, 1.0)
    obs[42] = min(state.runtime_hours[1] / 8760.0, 1.0)
    obs[43] = min(state.runtime_hours[2] / 8760.0, 1.0)
    return obs
