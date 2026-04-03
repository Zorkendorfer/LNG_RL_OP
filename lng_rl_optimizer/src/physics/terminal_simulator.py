import numpy as np
from dataclasses import dataclass, field
from .bog_model import BOGModel
from .compressor_model import BOGCompressorModel
from .thermodynamics import LNGComposition


@dataclass
class TerminalState:
    """Complete state of the LNG terminal at one timestep."""
    liquid_volume_m3: float
    tank_pressure_kPa: float
    lng_temp_K: float

    x_methane: float = 0.90
    x_ethane:  float = 0.07
    x_propane: float = 0.02
    x_nitrogen: float = 0.01

    n_compressors_running: int = 2
    compressor_loads: list = field(default_factory=lambda: [0.8, 0.8, 0.0])
    n_hp_pumps_running: int = 2
    n_vaporizers_running: int = 2

    ambient_temp_C: float = 10.0
    seawater_temp_C: float = 8.0
    sea_state: float = 0.0

    electricity_price_eur_mwh: float = 50.0
    price_forecast_24h: list = field(default_factory=lambda: [50.0] * 24)

    send_out_rate_t_h: float = 100.0
    bog_generation_kg_h: float = 0.0
    total_power_kW: float = 0.0
    electricity_cost_eur_h: float = 0.0

    compressor_starts_today: int = 0
    runtime_hours: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @property
    def fill_fraction(self) -> float:
        return self.liquid_volume_m3 / 155000.0

    @property
    def composition(self) -> LNGComposition:
        return LNGComposition(
            methane=self.x_methane,
            ethane=self.x_ethane,
            propane=self.x_propane,
            nitrogen=self.x_nitrogen,
        )


class LNGTerminalSimulator:
    """
    Physics-based ODE simulator for LNG terminal dynamics.
    Timestep: 1 hour.
    Used to generate PINN training data and for offline validation.
    NOT used during online RL training — PINN replaces it.
    """

    def __init__(self, config: dict):
        self.config = config
        self.bog_model  = BOGModel(config)
        self.compressor = BOGCompressorModel(config)

    def step(
        self,
        state: TerminalState,
        action: dict,
        dt_h: float = 1.0,
    ) -> tuple:
        """
        Advance terminal state by dt_h hours given control action.

        action keys:
            compressor_loads:  [0..1, 0..1, 0..1]
            n_hp_pumps:        int
            n_vaporizers:      int
            send_out_rate_t_h: float
        """
        import copy
        new_state = copy.deepcopy(state)

        bog_gen = self.bog_model.steady_state_bog(
            composition=state.composition,
            fill_level_fraction=state.fill_fraction,
            ambient_temp_C=state.ambient_temp_C,
            sea_state=state.sea_state,
        )

        total_comp_flow = 0.0
        comp_power = 0.0
        new_state.compressor_loads = list(action["compressor_loads"])

        for i, load in enumerate(action["compressor_loads"]):
            if load > 0.05:
                flow = load * self.compressor.max_capacity_kg_h
                power = self.compressor.power_kW(
                    flow_kg_h=flow,
                    suction_pressure_kPa=state.tank_pressure_kPa,
                    discharge_pressure_kPa=7000,
                    composition_methane_frac=state.x_methane,
                )
                total_comp_flow += flow
                comp_power += power
                new_state.runtime_hours[i] += dt_h
                if state.compressor_loads[i] <= 0.05:
                    new_state.compressor_starts_today += 1

        send_out_kg_h = action["send_out_rate_t_h"] * 1000

        n_hp = action["n_hp_pumps"]
        hp_pump_power = n_hp * self.config["hp_pumps"]["power_kW"][0]

        n_vap = action["n_vaporizers"]
        vap_power = n_vap * self.config["vaporizers"]["seawater_pump_kW"]

        lp_pump_power = 6 * self.config["lp_pumps"]["power_kW"][0] * 0.6

        total_power = comp_power + hp_pump_power + vap_power + lp_pump_power
        new_state.total_power_kW = total_power

        energy_kwh = total_power * dt_h
        new_state.electricity_cost_eur_h = (
            energy_kwh * state.electricity_price_eur_mwh / 1000
        )

        bog_recondensed = min(total_comp_flow, bog_gen * 1.1)
        net_bog_kg = (bog_gen - bog_recondensed) * dt_h

        lng_sent_kg = send_out_kg_h * dt_h
        rho = state.composition.liquid_density_kg_m3
        d_volume = -(lng_sent_kg / rho) - (net_bog_kg / rho)
        new_state.liquid_volume_m3 = np.clip(
            state.liquid_volume_m3 + d_volume,
            self.config["tanks"]["min_level_m3"],
            self.config["tanks"]["working_capacity_m3"],
        )

        vapor_volume = (
            self.config["tanks"]["total_capacity_m3"] - new_state.liquid_volume_m3
        )
        dp = net_bog_kg / (vapor_volume + 1e-6) * 0.5
        new_state.tank_pressure_kPa = np.clip(
            state.tank_pressure_kPa + dp,
            self.config["tanks"]["min_pressure_kPa"],
            self.config["tanks"]["max_pressure_kPa"],
        )

        evap_fraction = bog_gen * dt_h / (rho * state.liquid_volume_m3 + 1e-6)
        aging_rate = evap_fraction * 0.1
        new_state.x_methane  = max(0.80, state.x_methane - aging_rate * 0.5)
        new_state.x_ethane   = min(0.15, state.x_ethane  + aging_rate * 0.3)
        new_state.x_propane  = min(0.08, state.x_propane + aging_rate * 0.2)
        total = (new_state.x_methane + new_state.x_ethane +
                 new_state.x_propane + new_state.x_nitrogen)
        new_state.x_methane  /= total
        new_state.x_ethane   /= total
        new_state.x_propane  /= total
        new_state.x_nitrogen /= total

        new_state.bog_generation_kg_h = bog_gen
        new_state.send_out_rate_t_h   = action["send_out_rate_t_h"]
        new_state.n_hp_pumps_running  = action["n_hp_pumps"]
        new_state.n_vaporizers_running = action["n_vaporizers"]

        info = {
            "bog_gen_kg_h":     bog_gen,
            "bog_removed_kg_h": bog_recondensed,
            "comp_power_kW":    comp_power,
            "total_power_kW":   total_power,
            "cost_eur_h":       new_state.electricity_cost_eur_h,
            "tank_pressure":    new_state.tank_pressure_kPa,
            "fill_fraction":    new_state.fill_fraction,
        }

        return new_state, info
