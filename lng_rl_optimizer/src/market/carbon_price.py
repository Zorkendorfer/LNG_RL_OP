"""EU ETS carbon cost calculator."""


GRID_EMISSION_FACTOR_T_CO2_MWH = 0.30  # Lithuanian grid, 2024
CARBON_PRICE_EUR_T = 65.0              # approximate EU ETS price


def carbon_cost_eur_h(total_power_kW: float, dt_h: float = 1.0) -> float:
    """
    Calculate carbon cost for given power consumption.
    Returns EUR for the given time period.
    """
    energy_mwh = total_power_kW * dt_h / 1000
    co2_t      = energy_mwh * GRID_EMISSION_FACTOR_T_CO2_MWH
    return co2_t * CARBON_PRICE_EUR_T
