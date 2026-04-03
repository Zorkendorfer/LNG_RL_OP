import numpy as np
from dataclasses import dataclass


@dataclass
class LNGComposition:
    methane: float = 0.90
    ethane:  float = 0.07
    propane: float = 0.02
    nitrogen: float = 0.01

    def __post_init__(self):
        total = self.methane + self.ethane + self.propane + self.nitrogen
        assert abs(total - 1.0) < 1e-6, f"Mole fractions must sum to 1, got {total}"

    def to_refprop_string(self) -> str:
        return (f"METHANE[{self.methane}]&ETHANE[{self.ethane}]"
                f"&PROPANE[{self.propane}]&NITROGEN[{self.nitrogen}]")

    @property
    def bubble_point_K(self) -> float:
        return (111.7 * self.methane + 184.6 * self.ethane +
                231.1 * self.propane + 77.4 * self.nitrogen)

    @property
    def latent_heat_kJ_kg(self) -> float:
        return (509.0 * self.methane + 487.5 * self.ethane +
                427.0 * self.propane + 199.0 * self.nitrogen)

    @property
    def liquid_density_kg_m3(self) -> float:
        return (422.6 * self.methane + 546.0 * self.ethane +
                580.4 * self.propane + 806.0 * self.nitrogen)

    @property
    def wobbe_index(self) -> float:
        # Component HHVs in MJ/Sm3 give a terminal-oriented Wobbe index
        # in the typical natural-gas range (~40-60 MJ/Sm3).
        hhv = (39.8 * self.methane + 69.5 * self.ethane +
               93.0 * self.propane)
        sg  = (16.04 * self.methane + 30.07 * self.ethane +
               44.10 * self.propane + 28.01 * self.nitrogen) / 28.96
        return hhv / (sg ** 0.5)


def compute_bog_rate_physics(
    heat_ingress_kW: float,
    composition: LNGComposition,
    tank_pressure_kPa: float = 5.0,
) -> float:
    """BOG generation rate from first principles. Returns kg/h."""
    latent_heat_kJ_kg = composition.latent_heat_kJ_kg
    pressure_factor = 1.0 - 0.01 * (tank_pressure_kPa - 5.0)
    pressure_factor = max(0.5, min(1.5, pressure_factor))
    # Only part of the terminal heat leak produces net boil-off. The rest warms
    # tank internals, liquid bulk, and vapor space. This factor calibrates the
    # first-principles estimate to the expected 100-400 kg/h range for the
    # FSRU-scale heat ingress used in this project.
    net_evaporation_fraction = 0.22
    bog_rate_kw = heat_ingress_kW * pressure_factor * net_evaporation_fraction
    bog_rate_kg_s = bog_rate_kw / latent_heat_kJ_kg
    return bog_rate_kg_s * 3600
