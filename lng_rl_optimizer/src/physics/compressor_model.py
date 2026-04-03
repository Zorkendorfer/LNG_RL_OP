import numpy as np
from scipy.interpolate import interp1d


class BOGCompressorModel:
    """
    Reciprocating BOG compressor power model.
    Key insight: compressor power is NOT linear with flow.
    """

    def __init__(self, config: dict):
        cc = config["bog_compressors"]
        self.max_capacity_kg_h = cc["capacity_kg_per_h"][0]
        self.rated_power_kW    = cc["power_kW"][0]
        self.min_load          = cc["min_load_fraction"]

        self._load_points = [0.30, 0.50, 0.70, 0.80, 1.00]
        self._eff_points  = [0.65, 0.74, 0.79, 0.82, 0.80]
        self._eff_curve   = interp1d(
            self._load_points, self._eff_points,
            kind="cubic", fill_value="extrapolate"
        )

    def power_kW(
        self,
        flow_kg_h: float,
        suction_pressure_kPa: float,
        discharge_pressure_kPa: float,
        composition_methane_frac: float = 0.90,
    ) -> float:
        """Compute actual shaft power for given operating conditions."""
        if flow_kg_h <= 0:
            return 0.0

        load_fraction = flow_kg_h / self.max_capacity_kg_h
        load_fraction = np.clip(load_fraction, self.min_load, 1.0)

        gamma = 1.31 - 0.05 * (1 - composition_methane_frac)
        n     = gamma
        pr    = discharge_pressure_kPa / suction_pressure_kPa
        MW    = 16.04 * composition_methane_frac + 30.07 * (1 - composition_methane_frac)
        R_spec = 8314 / MW
        T_in   = 113.0

        polytropic_work_J_kg = (
            (n / (n - 1)) * R_spec * T_in * (pr ** ((n - 1) / n) - 1)
        )

        flow_kg_s = flow_kg_h / 3600
        isentropic_power_kW = flow_kg_s * polytropic_work_J_kg / 1000

        efficiency = float(self._eff_curve(load_fraction))
        return isentropic_power_kW / efficiency

    def max_flow_at_conditions(self, suction_pressure_kPa: float) -> float:
        """Maximum BOG flow at given suction pressure."""
        rho_ratio = suction_pressure_kPa / 101.325
        return self.max_capacity_kg_h * rho_ratio
