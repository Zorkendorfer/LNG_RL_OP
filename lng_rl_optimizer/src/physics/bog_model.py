import numpy as np
from .thermodynamics import LNGComposition, compute_bog_rate_physics


class BOGModel:
    """
    Physics model for BOG generation in LNG storage tanks.

    BOG sources:
    1. Steady heat ingress through insulation
    2. Flash evaporation during tank pressure changes
    3. Recirculation line heat input
    4. FSRU-specific: motion-induced mixing
    """

    def __init__(self, config: dict):
        self.base_heat_ingress_kW = config["tanks"]["heat_ingress_kW"]
        self.tank_volume_m3       = config["tanks"]["total_capacity_m3"]

    def steady_state_bog(
        self,
        composition: LNGComposition,
        fill_level_fraction: float,
        ambient_temp_C: float,
        sea_state: float = 0.0,
    ) -> float:
        """Steady-state BOG generation rate in kg/h."""
        area_factor    = 0.7 + 0.3 * fill_level_fraction
        ambient_factor = 1.0 + 0.02 * max(0, ambient_temp_C - 15)
        motion_factor  = 1.0 + 0.15 * sea_state

        total_heat_kW = (self.base_heat_ingress_kW
                        * area_factor * ambient_factor * motion_factor)
        return compute_bog_rate_physics(total_heat_kW, composition)

    def flash_bog(
        self,
        liquid_volume_m3: float,
        dp_kPa: float,
        composition: LNGComposition,
    ) -> float:
        """Flash BOG when tank pressure is reduced. Returns kg (instantaneous)."""
        if dp_kPa >= 0:
            return 0.0
        liquid_mass_kg = liquid_volume_m3 * composition.liquid_density_kg_m3
        flash_fraction = -dp_kPa * 0.001
        return liquid_mass_kg * flash_fraction
