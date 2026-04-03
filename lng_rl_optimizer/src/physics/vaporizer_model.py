import numpy as np


class ORVaporizerModel:
    """
    Open Rack Vaporizer (ORV) heat transfer model.
    Seawater-heated LNG vaporization.
    """

    def __init__(self, config: dict):
        vc = config["vaporizers"]
        self.capacity_t_per_h      = vc["capacity_t_per_h"][0]
        self.seawater_flow_m3_per_h = vc["seawater_flow_m3_per_h"]
        self.seawater_pump_kW       = vc["seawater_pump_kW"]
        self.min_seawater_temp_C    = vc["min_seawater_temp_C"]

    def can_operate(self, seawater_temp_C: float) -> bool:
        """ORV switches to SCV below minimum seawater temperature."""
        return seawater_temp_C >= self.min_seawater_temp_C

    def seawater_outlet_temp(
        self,
        lng_flow_t_per_h: float,
        seawater_temp_in_C: float,
    ) -> float:
        """
        Approximate seawater temperature drop across ORV panels.
        LNG requires ~500 kJ/kg to vaporize + heat to delivery temp.
        """
        lng_flow_kg_s = lng_flow_t_per_h * 1000 / 3600
        heat_load_kW  = lng_flow_kg_s * 550  # approx 550 kJ/kg for full vaporization
        seawater_mass_flow_kg_s = self.seawater_flow_m3_per_h * 1025 / 3600
        cp_seawater = 3.99  # kJ/(kg·K)
        dt = heat_load_kW / (seawater_mass_flow_kg_s * cp_seawater)
        return seawater_temp_in_C - dt
