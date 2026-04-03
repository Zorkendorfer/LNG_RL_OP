import numpy as np


class CentrifugalPumpModel:
    """
    Centrifugal pump power model for LNG LP and HP pumps.
    """

    def __init__(self, flow_m3_per_h: float, head_m: float, efficiency: float, power_kW: float):
        self.design_flow = flow_m3_per_h
        self.design_head = head_m
        self.efficiency  = efficiency
        self.rated_power = power_kW

    def power_at_flow(self, flow_fraction: float, rho_kg_m3: float = 430.0) -> float:
        """
        Power at given flow fraction (0–1) relative to design point.
        Uses affinity laws: P ∝ Q³ (ideally), modified for real pumps.
        """
        if flow_fraction <= 0:
            return 0.0
        flow_fraction = np.clip(flow_fraction, 0.1, 1.0)
        # Real pump: power curve is flatter than cubic at low flows
        power_fraction = 0.3 + 0.7 * flow_fraction ** 2.5
        return self.rated_power * power_fraction
