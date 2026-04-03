# Electricity-Price-Aware Safe RL for LNG Terminal Energy Optimization
# Implementation Plan — Claude Code Ready
>
> Novel contribution: physics-informed surrogate + RL agent that optimizes
> against real-time NordPool electricity prices with LNG composition as a
> state variable. Directly applicable to Klaipėda FSRU (Independence).
>
> Stack: Python 3.11+, CoolProp, PyTorch, Gymnasium, Stable-Baselines3, MLflow

---

## Research framing (include this in your thesis introduction)

**What exists:** RALT-DT (2025) uses PPO to optimize pump/vaporizer
scheduling for wear and load balance. MINLP models minimize total energy
volume consumed. HYSYS + PSO optimizes steady-state operating points.

**What this project adds (three genuine novelties):**
1. Optimization target is electricity *cost* not energy *volume* — requires
   coupling to real-time NordPool Baltic price forecasts. Fundamentally
   different problem when prices vary 10× within a day.
2. LNG composition (methane/ethane/propane fractions) is a dynamic state
   variable. Existing RL papers assume fixed composition. Klaipėda receives
   multi-source LNG with varying compositions.
3. World model uses physics-informed neural networks that encode energy
   balance and thermodynamic constraints as loss terms — not black-box
   neural networks. Gives physical plausibility and better generalization.

**Target venue:** Applied Energy, Energy, or Computers & Chemical Engineering.
All three have published LNG terminal optimization papers and would recognize
the novelty of the electricity-market coupling angle.

---

## Repository structure

```
lng_rl_optimizer/
├── CLAUDE.md                        # session context for Claude Code
├── README.md
├── pyproject.toml
├── .env.example                     # NORDPOOL_API_KEY (if needed)
│
├── config/
│   └── terminal.yaml                # FSRU parameters, equipment specs
│
├── data/
│   ├── nordpool/
│   │   ├── raw/                     # downloaded hourly price CSVs
│   │   └── processed/               # cleaned, filled, normalized
│   ├── weather/
│   │   └── klaipeda_historical.csv  # seawater temp, ambient temp, wind
│   ├── synthetic/
│   │   └── terminal_trajectories/   # surrogate training data from physics sim
│   └── composition/
│       └── lng_compositions.csv     # historical LNG cargo compositions
│
├── src/
│   ├── __init__.py
│   │
│   ├── physics/
│   │   ├── __init__.py
│   │   ├── thermodynamics.py        # LNG properties via CoolProp
│   │   ├── bog_model.py             # BOG generation rate physics model
│   │   ├── compressor_model.py      # reciprocating compressor power model
│   │   ├── pump_model.py            # centrifugal pump power model
│   │   ├── vaporizer_model.py       # ORV heat transfer model
│   │   └── terminal_simulator.py   # full terminal ODE system
│   │
│   ├── surrogate/
│   │   ├── __init__.py
│   │   ├── data_generator.py        # run physics sim → training dataset
│   │   ├── pinn.py                  # physics-informed neural network
│   │   ├── trainer.py               # PINN training loop
│   │   └── validator.py             # compare PINN vs physics model
│   │
│   ├── market/
│   │   ├── __init__.py
│   │   ├── nordpool_fetcher.py      # download historical + forecast prices
│   │   ├── price_forecaster.py      # LSTM for 24h ahead price prediction
│   │   └── carbon_price.py          # EU ETS carbon cost calculator
│   │
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── lng_terminal_env.py      # Gymnasium environment (core)
│   │   ├── state_space.py           # state definition + normalization
│   │   ├── action_space.py          # discrete action definitions
│   │   ├── reward.py                # cost-based reward function
│   │   └── safety_layer.py          # constraint projection layer
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py             # PPO with safety constraints (SB3)
│   │   ├── safe_ppo.py              # custom safe PPO with CBF layer
│   │   └── baseline_agents.py       # rule-based + MPC baselines
│   │
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py               # cost savings, constraint violations
│       ├── visualizer.py            # price vs action plots, cost breakdown
│       └── ablation.py              # remove each novelty, measure impact
│
├── scripts/
│   ├── check_env.py
│   ├── download_nordpool_data.py
│   ├── generate_surrogate_data.py
│   ├── train_surrogate.py
│   ├── train_price_forecaster.py
│   ├── train_agent.py
│   ├── evaluate_agent.py
│   └── run_ablation.py
│
├── notebooks/
│   ├── 01_nordpool_price_analysis.ipynb
│   ├── 02_terminal_physics_validation.ipynb
│   ├── 03_surrogate_accuracy.ipynb
│   └── 04_agent_cost_savings.ipynb
│
└── tests/
    ├── test_thermodynamics.py
    ├── test_bog_model.py
    ├── test_pinn.py
    ├── test_environment.py
    └── test_reward.py
```

---

## Phase 0 — Setup and data acquisition

### 0.1 Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate

# Thermodynamics
pip install CoolProp

# ML stack
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install stable-baselines3==2.2.1 gymnasium==0.29.1

# Data and analysis
pip install pandas numpy scipy matplotlib mlflow tqdm click pyyaml \
            requests scikit-learn jupyter ipykernel pytest

# Optional: for price forecasting
pip install statsmodels
```

### 0.2 pyproject.toml

```toml
[project]
name = "lng_rl_optimizer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "CoolProp>=6.6",
    "torch==2.1.0",
    "stable-baselines3==2.2.1",
    "gymnasium==0.29.1",
    "pandas>=2.1",
    "numpy>=1.26",
    "scipy>=1.11",
    "matplotlib>=3.8",
    "mlflow>=2.8",
    "tqdm>=4.66",
    "click>=8.1",
    "pyyaml>=6.0",
    "requests>=2.31",
    "scikit-learn>=1.3",
]
[project.optional-dependencies]
dev = ["pytest>=7.4", "jupyter>=1.0"]
```

### 0.3 Terminal configuration (`config/terminal.yaml`)

Model the Klaipėda FSRU Independence — publicly documented capacity.

```yaml
terminal:
  name: "Klaipėda FSRU Independence (model)"
  type: FSRU  # floating storage and regasification unit

tanks:
  n_tanks: 4                         # Independence has 4 membrane tanks
  total_capacity_m3: 170000          # m³ total LNG
  working_capacity_m3: 155000        # 90% fill limit (safety)
  min_level_m3: 17000                # 10% minimum (pump NPSH)
  design_pressure_kPa: 25.0          # gauge, tank operating limit
  max_pressure_kPa: 22.0             # operational maximum
  min_pressure_kPa: 2.0              # minimum before BOG deficit
  heat_ingress_kW: 180.0             # total heat leak at typical conditions
  # Note: FSRU adds motion-induced stratification — modeled as extra
  # 10-20% heat ingress during adverse sea states

bog_compressors:
  n_units: 3                         # typically 2 operating + 1 standby
  capacity_kg_per_h: [8000, 8000, 8000]
  power_kW: [1200, 1200, 1200]       # at full load, from compressor curves
  min_load_fraction: 0.3             # cannot run below 30%
  startup_time_min: 15               # time to bring from standby to load
  wear_cost_per_start: 500.0         # EUR, penalty for unnecessary cycling

lp_pumps:
  n_units: 6                         # in-tank pumps
  flow_m3_per_h: [300, 300, 300, 300, 300, 300]
  head_m: 50.0
  efficiency: 0.72
  power_kW: [55, 55, 55, 55, 55, 55]

hp_pumps:
  n_units: 4
  flow_m3_per_h: [150, 150, 150, 150]
  design_pressure_bar: 80.0
  power_kW: [450, 450, 450, 450]

vaporizers:
  type: ORV                          # open rack vaporizer (seawater)
  n_units: 4
  capacity_t_per_h: [100, 100, 100, 100]
  seawater_flow_m3_per_h: 3000       # per unit
  seawater_pump_kW: 185              # per unit
  min_seawater_temp_C: 4.0           # cutoff: switch to SCV below this
  fouling_factor_initial: 0.0        # degrades over time

send_out:
  max_rate_t_per_h: 400              # maximum pipeline send-out
  min_rate_t_per_h: 20               # minimum contractual send-out
  nominal_pressure_bar: 70.0

grid:
  voltage_kV: 110
  connection: "Klaipėda substation"
  max_import_MW: 15.0
```

### 0.4 Download NordPool Baltic price data

NordPool publishes historical hourly electricity prices for the Baltic
bidding zone (Lithuania) as free CSV files.

```python
# scripts/download_nordpool_data.py

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import click


NORDPOOL_BASE = "https://www.nordpoolgroup.com/api/marketdata/page/10"
# Note: NordPool API changed in 2024. Use the ENTSO-E Transparency Platform
# as backup — fully public, no API key required for historical data.
ENTSOE_BASE = "https://transparency.entsoe.eu/api"


def fetch_entsoe_prices(
    start: str,
    end: str,
    bidding_zone: str = "10YLT-1001A0008Q",  # Lithuania
    output_dir: Path = Path("data/nordpool/raw"),
) -> pd.DataFrame:
    """
    Fetch hourly Day-Ahead prices from ENTSO-E Transparency Platform.
    Free public API — register at transparency.entsoe.eu for an API token.
    Bidding zone code for Lithuania: 10YLT-1001A0008Q
    """
    # ENTSO-E requires registration but data is free
    # API token: set ENTSOE_TOKEN environment variable
    import os
    token = os.environ.get("ENTSOE_TOKEN")

    if not token:
        print(
            "No ENTSOE_TOKEN found. Register free at transparency.entsoe.eu\n"
            "Falling back to manual CSV download instructions:"
        )
        _print_manual_download_instructions()
        return pd.DataFrame()

    params = {
        "securityToken": token,
        "documentType":  "A44",      # Day-ahead prices
        "in_Domain":     bidding_zone,
        "out_Domain":    bidding_zone,
        "periodStart":   start,      # format: YYYYMMDDHHММ
        "periodEnd":     end,
    }

    response = requests.get(f"{ENTSOE_BASE}", params=params, timeout=30)
    # Parse XML response → DataFrame
    # Use the 'entsoe-py' package for clean parsing
    # pip install entsoe-py
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=token)
    prices = client.query_day_ahead_prices(
        "LT",
        start=pd.Timestamp(start, tz="Europe/Vilnius"),
        end=pd.Timestamp(end, tz="Europe/Vilnius"),
    )
    return prices


def _print_manual_download_instructions():
    print("""
Manual download (no API key required):
1. Go to https://transparency.entsoe.eu/
2. Market Data > Day-Ahead Prices > LT (Lithuania)
3. Select date range 2018-01-01 to 2024-12-31
4. Export as CSV
5. Save to data/nordpool/raw/lt_prices_YYYY.csv
""")


@click.command()
@click.option("--years", default="2020,2021,2022,2023,2024")
@click.option("--output-dir", default="data/nordpool/raw")
def download(years, output_dir):
    for year in years.split(","):
        start = f"{year}0101"
        end   = f"{year}1231"
        prices = fetch_entsoe_prices(start, end)
        if not prices.empty:
            path = Path(output_dir) / f"lt_prices_{year}.csv"
            prices.to_csv(path)
            print(f"Saved {year}: {len(prices)} hourly prices")
```

### 0.5 Price data preprocessor

```python
# src/market/nordpool_fetcher.py

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_clean_prices(data_dir: Path) -> pd.DataFrame:
    """
    Load all yearly CSVs, concatenate, clean, and return hourly price series.
    Returns DataFrame with columns: [timestamp, price_eur_mwh, day_of_week,
                                     hour, month, is_weekend, price_log]
    """
    dfs = []
    for csv_path in sorted(data_dir.glob("lt_prices_*.csv")):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        dfs.append(df)

    prices = pd.concat(dfs).sort_index()
    prices.index = prices.index.tz_convert("UTC")

    # Remove duplicates from DST transitions
    prices = prices[~prices.index.duplicated(keep="first")]

    # Fill gaps (< 4h) with forward fill; flag longer gaps
    n_missing = prices.isna().sum()
    if n_missing > 0:
        print(f"Filling {n_missing} missing hours")
        prices = prices.ffill(limit=4)

    # Add temporal features for price forecasting
    prices = prices.rename(columns={prices.columns[0]: "price_eur_mwh"})
    prices["hour"]       = prices.index.hour
    prices["day_of_week"] = prices.index.dayofweek
    prices["month"]      = prices.index.month
    prices["is_weekend"] = prices.index.dayofweek >= 5
    prices["price_log"]  = np.log1p(prices["price_eur_mwh"].clip(lower=0))

    print(f"Price range: €{prices.price_eur_mwh.min():.1f} – "
          f"€{prices.price_eur_mwh.max():.1f}/MWh")
    print(f"Mean: €{prices.price_eur_mwh.mean():.1f}/MWh, "
          f"Std: €{prices.price_eur_mwh.std():.1f}/MWh")

    return prices
```

**Milestone:** `python scripts/download_nordpool_data.py` produces CSV files.
Load them: `df = load_and_clean_prices(Path("data/nordpool/raw"))`.
Plot price distributions by hour of day in notebook 01.

---

## Phase 1 — Physics model

This is the foundation. The physics must be correct before the PINN or RL.
Do not skip validation of each component.

### 1.1 Thermodynamic properties (`src/physics/thermodynamics.py`)

```python
import numpy as np
from CoolProp.CoolProp import PropsSI
from dataclasses import dataclass


# LNG composition: mole fractions summing to 1.0
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
        """CoolProp mixture string for thermodynamic calculations."""
        return (f"METHANE[{self.methane}]&ETHANE[{self.ethane}]"
                f"&PROPANE[{self.propane}]&NITROGEN[{self.nitrogen}]")

    @property
    def bubble_point_K(self) -> float:
        """Approximate bubble point temperature at 1 atm."""
        # Linear interpolation (accurate to ~1K for typical LNG)
        return (111.7 * self.methane + 184.6 * self.ethane +
                231.1 * self.propane + 77.4 * self.nitrogen)

    @property
    def latent_heat_kJ_kg(self) -> float:
        """Approximate latent heat of vaporization at bubble point."""
        return (509.0 * self.methane + 487.5 * self.ethane +
                427.0 * self.propane + 199.0 * self.nitrogen)

    @property
    def liquid_density_kg_m3(self) -> float:
        """Approximate liquid density at bubble point, 1 atm."""
        return (422.6 * self.methane + 546.0 * self.ethane +
                580.4 * self.propane + 806.0 * self.nitrogen)

    @property
    def wobbe_index(self) -> float:
        """
        Wobbe index — measure of interchangeability for gas turbines.
        Higher ethane/propane = higher calorific value = higher Wobbe.
        Used to check if send-out NG meets pipeline spec.
        """
        hhv = (890.4 * self.methane + 1559.8 * self.ethane +
               2220.0 * self.propane) / 1000  # MJ/mol
        sg  = (16.04 * self.methane + 30.07 * self.ethane +
               44.10 * self.propane + 28.01 * self.nitrogen) / 28.96
        return hhv / (sg ** 0.5)


def compute_bog_rate_physics(
    heat_ingress_kW: float,
    composition: LNGComposition,
    tank_pressure_kPa: float = 5.0,
) -> float:
    """
    BOG generation rate from first principles.
    BOG rate = heat_ingress / latent_heat_of_vaporization

    Returns: BOG rate in kg/h
    """
    latent_heat_kJ_kg = composition.latent_heat_kJ_kg
    # Pressure correction: higher pressure = higher boiling point = less BOG
    pressure_factor = 1.0 - 0.01 * (tank_pressure_kPa - 5.0)
    pressure_factor = max(0.5, min(1.5, pressure_factor))

    bog_rate_kw = heat_ingress_kW * pressure_factor
    bog_rate_kg_s = bog_rate_kw / latent_heat_kJ_kg
    return bog_rate_kg_s * 3600  # kg/h
```

### 1.2 BOG model (`src/physics/bog_model.py`)

```python
import numpy as np
from .thermodynamics import LNGComposition, compute_bog_rate_physics


class BOGModel:
    """
    Physics model for BOG generation in LNG storage tanks.

    BOG sources:
    1. Steady heat ingress through insulation (dominant in holding mode)
    2. Flash evaporation during tank pressure changes
    3. Recirculation line heat input (LNG returned from pumps/pipes)
    4. FSRU-specific: motion-induced mixing (increases effective heat ingress)

    BOG consumption:
    1. BOG compressors → recondenser → send-out
    2. Fuel gas system (for FSRU propulsion and hotel load)
    """

    def __init__(self, config: dict):
        self.base_heat_ingress_kW = config["tanks"]["heat_ingress_kW"]
        self.tank_volume_m3       = config["tanks"]["total_capacity_m3"]

    def steady_state_bog(
        self,
        composition: LNGComposition,
        fill_level_fraction: float,      # 0 to 1
        ambient_temp_C: float,
        sea_state: float = 0.0,          # 0=calm, 1=rough — FSRU specific
    ) -> float:
        """
        Steady-state BOG generation rate in kg/h.
        """
        # Heat ingress scales with surface area (∝ fill level for cylindrical tank)
        # Full tank has less vapor space → less heat into vapor
        area_factor = 0.7 + 0.3 * fill_level_fraction

        # Ambient temperature effect
        ambient_factor = 1.0 + 0.02 * max(0, ambient_temp_C - 15)

        # FSRU motion effect: wave-induced sloshing increases stratification
        # and effective heat transfer coefficient
        motion_factor = 1.0 + 0.15 * sea_state

        total_heat_kW = (self.base_heat_ingress_kW
                        * area_factor
                        * ambient_factor
                        * motion_factor)

        return compute_bog_rate_physics(total_heat_kW, composition)

    def flash_bog(
        self,
        liquid_volume_m3: float,
        dp_kPa: float,                   # pressure drop (negative = drop)
        composition: LNGComposition,
    ) -> float:
        """
        Flash BOG when tank pressure is reduced (e.g. by opening vent).
        Only relevant for sudden pressure changes.
        Returns additional BOG in kg/h (instantaneous).
        """
        if dp_kPa >= 0:
            return 0.0

        # Adiabatic flash: dP leads to some liquid vaporizing
        # Approximate: 0.1% of liquid mass per kPa pressure drop
        liquid_mass_kg = liquid_volume_m3 * composition.liquid_density_kg_m3
        flash_fraction = -dp_kPa * 0.001
        return liquid_mass_kg * flash_fraction   # kg (instantaneous, not /h)
```

### 1.3 Compressor model (`src/physics/compressor_model.py`)

```python
import numpy as np
from scipy.interpolate import interp1d


class BOGCompressorModel:
    """
    Reciprocating BOG compressor power model.
    Based on actual performance curves (polytropic compression).

    Key insight: compressor power is NOT linear with flow.
    Real power curve has a minimum at partial load — running at 50%
    capacity often uses 70% of full-load power. This non-linearity
    is why the MINLP models (and standard RL) get sub-optimal solutions.
    """

    def __init__(self, config: dict):
        # From config: compressor specs
        cc = config["bog_compressors"]
        self.max_capacity_kg_h = cc["capacity_kg_per_h"][0]
        self.rated_power_kW    = cc["power_kW"][0]
        self.min_load          = cc["min_load_fraction"]

        # Polytropic efficiency curve (load_fraction → efficiency)
        # Typical reciprocating compressor: peaks at 80% load
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
        """
        Compute actual shaft power for given operating conditions.
        Uses polytropic compression with real efficiency curve.
        """
        if flow_kg_h <= 0:
            return 0.0

        load_fraction = flow_kg_h / self.max_capacity_kg_h
        load_fraction = np.clip(load_fraction, self.min_load, 1.0)

        # Polytropic work per unit mass (isentropic approximation)
        # gamma ≈ 1.31 for methane-rich BOG
        gamma = 1.31 - 0.05 * (1 - composition_methane_frac)
        n     = gamma  # polytropic index ≈ gamma for reciprocating
        pr    = discharge_pressure_kPa / suction_pressure_kPa

        # Molecular weight ≈ 16.4 kg/kmol for typical LNG BOG
        MW    = 16.04 * composition_methane_frac + 30.07 * (1 - composition_methane_frac)
        R_spec = 8314 / MW  # J/(kg·K)
        T_in   = 113.0      # K, approximate BOG temperature at compressor inlet

        polytropic_work_J_kg = (
            (n / (n - 1)) * R_spec * T_in * (pr ** ((n - 1) / n) - 1)
        )

        # Convert to kW
        flow_kg_s = flow_kg_h / 3600
        isentropic_power_kW = flow_kg_s * polytropic_work_J_kg / 1000

        efficiency = float(self._eff_curve(load_fraction))
        return isentropic_power_kW / efficiency

    def max_flow_at_conditions(
        self,
        suction_pressure_kPa: float,
    ) -> float:
        """
        Maximum BOG flow at given suction pressure.
        Reciprocating compressors are approximately pressure-independent
        for flow (unlike centrifugal), but suction pressure affects
        actual mass flow via density.
        """
        rho_ratio = suction_pressure_kPa / 101.325  # relative to atmospheric
        return self.max_capacity_kg_h * rho_ratio
```

### 1.4 Full terminal simulator (`src/physics/terminal_simulator.py`)

```python
import numpy as np
from dataclasses import dataclass, field
from .bog_model import BOGModel
from .compressor_model import BOGCompressorModel
from .thermodynamics import LNGComposition


@dataclass
class TerminalState:
    """Complete state of the LNG terminal at one timestep."""
    # Tank
    liquid_volume_m3: float       # current LNG inventory
    tank_pressure_kPa: float      # vapor space pressure
    lng_temp_K: float             # bulk liquid temperature

    # Composition (mole fractions)
    x_methane: float = 0.90
    x_ethane:  float = 0.07
    x_propane: float = 0.02
    x_nitrogen: float = 0.01

    # Equipment state
    n_compressors_running: int = 2
    compressor_loads: list = field(default_factory=lambda: [0.8, 0.8, 0.0])
    n_hp_pumps_running: int = 2
    n_vaporizers_running: int = 2

    # External conditions
    ambient_temp_C: float = 10.0
    seawater_temp_C: float = 8.0
    sea_state: float = 0.0         # 0=calm, 1=rough

    # Market
    electricity_price_eur_mwh: float = 50.0
    price_forecast_24h: list = field(default_factory=lambda: [50.0] * 24)

    # Performance
    send_out_rate_t_h: float = 100.0
    bog_generation_kg_h: float = 0.0
    total_power_kW: float = 0.0
    electricity_cost_eur_h: float = 0.0

    # Wear tracking
    compressor_starts_today: int = 0
    runtime_hours: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @property
    def fill_fraction(self) -> float:
        return self.liquid_volume_m3 / 155000.0  # working capacity

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
    Timestep: 1 hour (matches electricity market resolution).

    This simulator is used to:
    1. Generate training data for the PINN surrogate
    2. Validate the PINN predictions
    3. Serve as the "true environment" for offline RL evaluation

    NOT used during online RL training (too slow) — PINN replaces it.
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
    ) -> tuple[TerminalState, dict]:
        """
        Advance terminal state by dt_h hours given control action.

        action dict keys:
            compressor_loads:  [0..1, 0..1, 0..1] — load fraction per unit
            n_hp_pumps:        int — number of HP pumps running
            n_vaporizers:      int — number of vaporizers running
            send_out_rate_t_h: float — target NG send-out rate

        Returns: (new_state, info_dict)
        """
        import copy
        new_state = copy.deepcopy(state)

        # --- BOG generation ---
        bog_gen = self.bog_model.steady_state_bog(
            composition=state.composition,
            fill_level_fraction=state.fill_fraction,
            ambient_temp_C=state.ambient_temp_C,
            sea_state=state.sea_state,
        )

        # --- Compressor operation ---
        total_comp_flow = 0.0
        comp_power = 0.0
        new_state.compressor_loads = action["compressor_loads"]

        for i, load in enumerate(action["compressor_loads"]):
            if load > 0.05:  # compressor is running
                flow = load * self.compressor.max_capacity_kg_h
                power = self.compressor.power_kW(
                    flow_kg_h=flow,
                    suction_pressure_kPa=state.tank_pressure_kPa,
                    discharge_pressure_kPa=7000,  # ~70 bar discharge
                    composition_methane_frac=state.x_methane,
                )
                total_comp_flow += flow
                comp_power += power
                new_state.runtime_hours[i] += dt_h

                # Track starts (load going from 0 → positive)
                if state.compressor_loads[i] <= 0.05:
                    new_state.compressor_starts_today += 1

        # --- Send-out (HP pumps + vaporizers) ---
        send_out_kg_h = action["send_out_rate_t_h"] * 1000  # t/h → kg/h

        # HP pump power
        n_hp = action["n_hp_pumps"]
        hp_pump_power = n_hp * self.config["hp_pumps"]["power_kW"][0]

        # ORV seawater pump power
        n_vap = action["n_vaporizers"]
        vap_power = n_vap * self.config["vaporizers"]["seawater_pump_kW"]

        # LP pump power
        lp_pump_power = 6 * self.config["lp_pumps"]["power_kW"][0] * 0.6

        # Total electrical power
        total_power = comp_power + hp_pump_power + vap_power + lp_pump_power
        new_state.total_power_kW = total_power

        # --- Electricity cost ---
        energy_kwh = total_power * dt_h
        new_state.electricity_cost_eur_h = (
            energy_kwh * state.electricity_price_eur_mwh / 1000
        )

        # --- Mass balance ---
        # BOG generated - BOG compressed (recondensed) = net BOG to tank vapor
        bog_recondensed = min(total_comp_flow, bog_gen * 1.1)  # can't condense more than generated
        net_bog_kg = (bog_gen - bog_recondensed) * dt_h

        # LNG mass sent out
        lng_sent_kg = send_out_kg_h * dt_h

        # Update inventory
        rho = state.composition.liquid_density_kg_m3
        d_volume = -(lng_sent_kg / rho) - (net_bog_kg / rho)
        new_state.liquid_volume_m3 = np.clip(
            state.liquid_volume_m3 + d_volume,
            self.config["tanks"]["min_level_m3"],
            self.config["tanks"]["working_capacity_m3"],
        )

        # --- Pressure update (simplified) ---
        # Net BOG in vapor space → pressure change
        vapor_volume = (
            self.config["tanks"]["total_capacity_m3"] - new_state.liquid_volume_m3
        )
        dp = net_bog_kg / vapor_volume * 0.5  # simplified, bar → kPa
        new_state.tank_pressure_kPa = np.clip(
            state.tank_pressure_kPa + dp,
            self.config["tanks"]["min_pressure_kPa"],
            self.config["tanks"]["max_pressure_kPa"],
        )

        # --- Composition aging ---
        # Preferential evaporation: methane evaporates faster than ethane
        # Remaining liquid gets slightly richer in heavier components over time
        evap_fraction = bog_gen * dt_h / (rho * state.liquid_volume_m3 + 1e-6)
        aging_rate = evap_fraction * 0.1  # small effect per timestep
        new_state.x_methane  = max(0.80, state.x_methane - aging_rate * 0.5)
        new_state.x_ethane   = min(0.15, state.x_ethane  + aging_rate * 0.3)
        new_state.x_propane  = min(0.08, state.x_propane + aging_rate * 0.2)
        # Renormalize
        total = new_state.x_methane + new_state.x_ethane + new_state.x_propane + new_state.x_nitrogen
        new_state.x_methane  /= total
        new_state.x_ethane   /= total
        new_state.x_propane  /= total
        new_state.x_nitrogen /= total

        # Record
        new_state.bog_generation_kg_h = bog_gen
        new_state.send_out_rate_t_h   = action["send_out_rate_t_h"]

        info = {
            "bog_gen_kg_h":    bog_gen,
            "bog_removed_kg_h": bog_recondensed,
            "comp_power_kW":   comp_power,
            "total_power_kW":  total_power,
            "cost_eur_h":      new_state.electricity_cost_eur_h,
            "tank_pressure":   new_state.tank_pressure_kPa,
            "fill_fraction":   new_state.fill_fraction,
        }

        return new_state, info
```

**Milestone:** run `notebooks/02_terminal_physics_validation.ipynb`. Simulate
24 hours of holding mode. Verify:
- BOG rate ~180–220 kg/h at typical conditions
- Tank pressure stable when 1 compressor runs at 60% load
- Power consumption ~1.5–2.5 MW total for nominal operation
- Compare BOG rate to published literature values for 170,000 m³ FSRU

---

## Phase 2 — Physics-informed surrogate (PINN)

The PINN replaces the slow physics simulator during RL training.
It must be fast (< 1ms per step), accurate (< 3% error), and physically
consistent (output must satisfy mass/energy balance).

### 2.1 Training data generation (`src/surrogate/data_generator.py`)

```python
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.physics.terminal_simulator import LNGTerminalSimulator, TerminalState
import yaml, json
from pathlib import Path


def generate_training_trajectories(
    n_episodes: int,
    episode_length_h: int,
    config: dict,
    output_dir: Path,
    seed: int = 42,
) -> None:
    """
    Generate diverse trajectories by running the physics simulator
    with random control actions and random initial conditions.

    Each trajectory captures dynamics that the PINN must learn:
    - BOG generation under varying conditions
    - Compressor power at different loads and compositions
    - Pressure evolution under different control strategies

    Saves each episode as a parquet file with columns:
    [hour, all state variables, all action variables, all outputs]
    """
    rng = np.random.default_rng(seed)
    sim = LNGTerminalSimulator(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode in tqdm(range(n_episodes), desc="Generating trajectories"):
        # Random initial state
        state = TerminalState(
            liquid_volume_m3   = rng.uniform(30000, 155000),
            tank_pressure_kPa  = rng.uniform(3, 20),
            lng_temp_K         = rng.uniform(110, 115),
            x_methane          = rng.uniform(0.82, 0.95),
            x_ethane           = rng.uniform(0.03, 0.12),
            x_propane          = rng.uniform(0.01, 0.05),
            x_nitrogen         = 0.01,
            ambient_temp_C     = rng.uniform(-10, 30),
            seawater_temp_C    = rng.uniform(2, 22),
            sea_state          = rng.uniform(0, 1),
            electricity_price_eur_mwh = rng.uniform(0, 300),
        )
        # Normalize composition
        total = state.x_methane + state.x_ethane + state.x_propane + state.x_nitrogen
        state.x_methane /= total
        state.x_ethane  /= total
        state.x_propane /= total
        state.x_nitrogen /= total

        records = []
        for h in range(episode_length_h):
            # Random action (exploration)
            action = {
                "compressor_loads": [
                    rng.choice([0.0, rng.uniform(0.3, 1.0)]),
                    rng.choice([0.0, rng.uniform(0.3, 1.0)]),
                    rng.choice([0.0, 0.0, 0.0, rng.uniform(0.3, 0.8)]),
                ],
                "n_hp_pumps":      int(rng.choice([1, 2, 3, 4])),
                "n_vaporizers":    int(rng.choice([1, 2, 3, 4])),
                "send_out_rate_t_h": float(rng.uniform(20, 380)),
            }

            new_state, info = sim.step(state, action)

            record = {
                "episode": episode, "hour": h,
                # Inputs (state + action)
                "fill_fraction":    state.fill_fraction,
                "tank_pressure_kPa": state.tank_pressure_kPa,
                "x_methane":        state.x_methane,
                "x_ethane":         state.x_ethane,
                "ambient_temp_C":   state.ambient_temp_C,
                "seawater_temp_C":  state.seawater_temp_C,
                "sea_state":        state.sea_state,
                "price_eur_mwh":    state.electricity_price_eur_mwh,
                "comp_load_0":      action["compressor_loads"][0],
                "comp_load_1":      action["compressor_loads"][1],
                "comp_load_2":      action["compressor_loads"][2],
                "n_hp_pumps":       action["n_hp_pumps"],
                "n_vaporizers":     action["n_vaporizers"],
                "send_out_rate":    action["send_out_rate_t_h"],
                # Outputs (targets for PINN)
                "bog_gen_kg_h":     info["bog_gen_kg_h"],
                "comp_power_kW":    info["comp_power_kW"],
                "total_power_kW":   info["total_power_kW"],
                "cost_eur_h":       info["cost_eur_h"],
                "new_pressure_kPa": new_state.tank_pressure_kPa,
                "new_fill_fraction": new_state.fill_fraction,
            }
            records.append(record)
            state = new_state

        df = pd.DataFrame(records)
        df.to_parquet(output_dir / f"episode_{episode:05d}.parquet")
```

### 2.2 Physics-informed neural network (`src/surrogate/pinn.py`)

```python
import torch
import torch.nn as nn
from torch import Tensor


class TerminalPINN(nn.Module):
    """
    Physics-Informed Neural Network for LNG terminal dynamics.

    Inputs (14 features):
        State:  fill_fraction, tank_pressure_kPa, x_methane, x_ethane,
                ambient_temp_C, seawater_temp_C, sea_state, price_eur_mwh
        Action: comp_load_0, comp_load_1, comp_load_2,
                n_hp_pumps (normalized), n_vaporizers (normalized),
                send_out_rate (normalized)

    Outputs (6):
        bog_gen_kg_h     — BOG generation rate
        comp_power_kW    — compressor electrical power
        total_power_kW   — all equipment power
        cost_eur_h       — electricity cost per hour
        new_pressure     — tank pressure after timestep
        new_fill         — fill fraction after timestep

    Physics constraints encoded in loss:
        1. total_power >= comp_power (can't use less than compressors alone)
        2. cost = total_power * price / 1000 (energy cost identity)
        3. fill rate consistency with send-out and BOG
        4. pressure change consistent with net BOG balance
        5. comp_power >= 0, bog_gen > 0 always (non-negativity)
    """

    def __init__(
        self,
        input_dim:  int = 14,
        hidden_dim: int = 256,
        n_layers:   int = 6,
        output_dim: int = 6,
        dropout:    float = 0.05,
    ):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, output_dim)

        # Output scaling (initialized to physics-based ranges)
        # Helps training: outputs are O(1) after scaling
        self.register_buffer("out_mean", torch.tensor([
            200.0,   # bog_gen_kg_h
            1500.0,  # comp_power_kW
            2500.0,  # total_power_kW
            125.0,   # cost_eur_h
            10.0,    # pressure_kPa
            0.5,     # fill_fraction
        ]))
        self.register_buffer("out_std", torch.tensor([
            150.0, 800.0, 1200.0, 100.0, 5.0, 0.2
        ]))

    def forward(self, x: Tensor) -> Tensor:
        """Returns normalized predictions. Use decode() for physical units."""
        h = self.backbone(x)
        return self.head(h)

    def predict_physical(self, x: Tensor) -> Tensor:
        """Returns predictions in physical units with non-negativity."""
        raw = self.forward(x)
        physical = raw * self.out_std + self.out_mean
        # Non-negativity constraints
        physical = torch.clamp(physical, min=torch.zeros(6, device=x.device))
        return physical


def physics_consistency_loss(
    pred: Tensor,
    inputs: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """
    Soft physics constraints encoded as additional loss terms.
    These regularize the network toward physically plausible outputs
    even in regions with sparse training data — the key PINN advantage.

    pred columns: [bog_gen, comp_power, total_power, cost, new_pressure, new_fill]
    inputs columns: [..., price_eur_mwh (idx 7), send_out_rate_norm (idx 13)]
    """
    bog_gen     = pred[:, 0]
    comp_power  = pred[:, 1]
    total_power = pred[:, 2]
    cost        = pred[:, 3]
    price       = inputs[:, 7]           # EUR/MWh
    send_out    = inputs[:, 13] * 400    # denormalize send-out rate

    # Constraint 1: total power >= compressor power
    # Violation if total_power < comp_power
    c1 = torch.relu(comp_power - total_power)

    # Constraint 2: cost = total_power (kW) * price (EUR/MWh) / 1000
    cost_physics = total_power * price / 1000  # EUR/h
    c2 = (cost - cost_physics).pow(2)

    # Constraint 3: BOG must be positive
    c3 = torch.relu(-bog_gen)

    # Constraint 4: comp power must be non-negative
    c4 = torch.relu(-comp_power)

    # Constraint 5: send-out > 0 requires pumps + vaporizers to be running
    # (proxy: total power > some minimum when send-out > 0)
    min_power_when_running = 200.0  # kW (at least one pump + vaporizer)
    c5 = torch.relu(
        (send_out > 10).float() * (min_power_when_running - total_power)
    )

    return weight * (c1.mean() + c2.mean() + c3.mean() + c4.mean() + c5.mean())
```

### 2.3 Surrogate trainer (`src/surrogate/trainer.py`)

```python
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pandas as pd
import numpy as np
from .pinn import TerminalPINN, physics_consistency_loss


def train_surrogate(
    data_dir: Path,
    output_dir: Path,
    hidden_dim: int = 256,
    n_layers: int = 6,
    lr: float = 3e-4,
    epochs: int = 200,
    batch_size: int = 1024,
    physics_weight: float = 0.5,
    device: str = "cuda",
) -> TerminalPINN:

    # Load all trajectories
    dfs = [pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))]
    df  = pd.concat(dfs, ignore_index=True)

    INPUT_COLS = [
        "fill_fraction", "tank_pressure_kPa", "x_methane", "x_ethane",
        "ambient_temp_C", "seawater_temp_C", "sea_state", "price_eur_mwh",
        "comp_load_0", "comp_load_1", "comp_load_2",
        "n_hp_pumps", "n_vaporizers", "send_out_rate",
    ]
    OUTPUT_COLS = [
        "bog_gen_kg_h", "comp_power_kW", "total_power_kW",
        "cost_eur_h", "new_pressure_kPa", "new_fill_fraction",
    ]

    # Normalize inputs
    x_mean = df[INPUT_COLS].mean()
    x_std  = df[INPUT_COLS].std().clip(lower=1e-6)
    X = torch.tensor(
        ((df[INPUT_COLS] - x_mean) / x_std).values,
        dtype=torch.float32
    )
    Y = torch.tensor(df[OUTPUT_COLS].values, dtype=torch.float32)

    # Train/val split
    n = len(X)
    n_train = int(n * 0.85)
    idx = torch.randperm(n)
    X_train, Y_train = X[idx[:n_train]], Y[idx[:n_train]]
    X_val,   Y_val   = X[idx[n_train:]], Y[idx[n_train:]]

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size, shuffle=True
    )

    model = TerminalPINN(hidden_dim=hidden_dim, n_layers=n_layers)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Normalize outputs against model's built-in scaling
    Y_train_norm = (Y_train - model.out_mean.cpu()) / model.out_std.cpu()
    Y_val_norm   = (Y_val   - model.out_mean.cpu()) / model.out_std.cpu()

    best_val_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="pinn_surrogate"):
        mlflow.log_params({
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "physics_weight": physics_weight,
            "n_train": n_train,
        })

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                yb_norm = (yb - model.out_mean) / model.out_std

                pred = model(xb)
                data_loss    = nn.functional.mse_loss(pred, yb_norm)
                physics_loss = physics_consistency_loss(
                    model.predict_physical(xb), xb, weight=physics_weight
                )
                loss = data_loss + physics_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                X_v = X_val.to(device)
                Y_v_norm = Y_val_norm.to(device)
                val_pred = model(X_v)
                val_loss = nn.functional.mse_loss(val_pred, Y_v_norm).item()

                # Relative L2 in physical units
                pred_phys = model.predict_physical(X_v).cpu()
                rel_l2 = (
                    torch.norm(pred_phys - Y_val) /
                    (torch.norm(Y_val) + 1e-8)
                ).item()

            mlflow.log_metrics({
                "train_loss": train_loss / len(train_loader),
                "val_loss":   val_loss,
                "val_rel_l2": rel_l2,
            }, step=epoch)

            if epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | "
                      f"val_rel_l2={rel_l2:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state": model.state_dict(),
                    "x_mean": x_mean.values,
                    "x_std":  x_std.values,
                    "input_cols": INPUT_COLS,
                }, output_dir / "best_pinn.pt")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Target: val_rel_l2 < 0.03 (3% error)")
    return model
```

**Milestone:** PINN achieves < 3% relative L2 error on held-out trajectories.
Check per-output errors in notebook 03 — BOG prediction and cost prediction
are the most critical. Cost error > 5% means the RL reward signal is too noisy.

---

## Phase 3 — Electricity price forecaster

### 3.1 24-hour ahead price forecaster (`src/market/price_forecaster.py`)

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class PriceForecaster(nn.Module):
    """
    LSTM-based 24-hour ahead electricity price forecaster.
    Input: last 168 hours (1 week) of prices + calendar features
    Output: 24-hour price forecast [EUR/MWh]

    This gives the RL agent predictive information about when
    electricity will be cheap — enabling load shifting.

    Simple architecture deliberately — forecasting is not the
    research contribution here, just an enabling component.
    """

    def __init__(
        self,
        input_dim:  int = 6,   # price, hour, dow, month, is_weekend, price_log
        hidden_dim: int = 128,
        n_layers:   int = 2,
        horizon:    int = 24,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers,
            batch_first=True, dropout=0.1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim) — price history
        returns: (batch, 24) — price forecast
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last hidden state → forecast


def prepare_price_features(
    prices_df: pd.DataFrame,
    lookback_h: int = 168,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare (X, y) pairs for price forecaster training.
    X: (n_samples, lookback_h, 6)
    y: (n_samples, 24)
    """
    price = prices_df["price_eur_mwh"].values
    price_log = np.log1p(np.clip(price, 0, None))
    hour = prices_df["hour"].values / 23.0
    dow  = prices_df["day_of_week"].values / 6.0
    month = (prices_df["month"].values - 1) / 11.0
    weekend = prices_df["is_weekend"].values.astype(float)
    # Normalize price to [0,1] approximately
    price_norm = np.clip(price / 300.0, 0, 1)

    features = np.stack([price_norm, price_log/6, hour, dow, month, weekend], axis=1)

    X, y = [], []
    for i in range(lookback_h, len(price) - 24):
        X.append(features[i-lookback_h:i])
        y.append(price[i:i+24] / 300.0)  # normalize targets

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
```

---

## Phase 4 — Gymnasium environment

### 4.1 State and action space (`src/environment/state_space.py`)

```python
import numpy as np
import gymnasium as gym


# Full state vector (26 dimensions):
# [0]   fill_fraction                (0–1)
# [1]   tank_pressure_norm           (0–1, mapped from 0–25 kPa)
# [2]   x_methane                    (0.80–0.95)
# [3]   x_ethane                     (0.03–0.15)
# [4]   ambient_temp_norm            (-1 to 1, mapped -20°C to 35°C)
# [5]   seawater_temp_norm           (0–1, mapped 2°C to 22°C)
# [6]   sea_state                    (0–1)
# [7]   current_price_norm           (0–1, mapped 0–300 EUR/MWh)
# [8-31] price_forecast_24h_norm     (24 values, 0–1 each)
# [32]   hour_of_day_sin              (seasonal encoding)
# [33]   hour_of_day_cos
# [34]   day_of_week_norm             (0–1)
# [35]   comp_load_0                  (0–1, current state)
# [36]   comp_load_1
# [37]   comp_load_2
# [38]   n_hp_pumps_norm              (0–1, mapped 0–4)
# [39]   n_vaporizers_norm            (0–1, mapped 0–4)
# [40]   compressor_starts_today_norm (0–1, mapped 0–10)
# [41]   runtime_hours_0_norm         (0–1, mapped 0–8760)
# [42]   runtime_hours_1_norm
# [43]   runtime_hours_2_norm

STATE_DIM = 44


def encode_state(state, price_forecast_24h: np.ndarray) -> np.ndarray:
    h = state.fill_fraction

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
    hour = state.fill_fraction  # placeholder — pass actual hour
    obs[32] = np.sin(2 * np.pi * 0 / 24)   # fill with actual hour
    obs[33] = np.cos(2 * np.pi * 0 / 24)
    obs[34] = 0.0  # day of week
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
```

### 4.2 Action space (`src/environment/action_space.py`)

```python
# Discrete action space — 5 decisions, each with a few levels
# Combined into one MultiDiscrete space for SB3 compatibility

# Action dimensions:
# [0] comp_0_load: 0=off, 1=30%, 2=60%, 3=100%      (4 choices)
# [1] comp_1_load: 0=off, 1=30%, 2=60%, 3=100%      (4 choices)
# [2] comp_2_load: 0=off, 1=30%, 2=60%               (3 choices, standby)
# [3] n_hp_pumps:  1, 2, 3, 4                        (4 choices)
# [4] n_vaporizers: 1, 2, 3, 4                       (4 choices)

COMP_LOADS = [0.0, 0.3, 0.6, 1.0]
STANDBY_LOADS = [0.0, 0.3, 0.6]
ACTION_DIMS = [4, 4, 3, 4, 4]   # MultiDiscrete shape

def decode_action(action_array: list, send_out_demand: float) -> dict:
    """Convert discrete action indices to physical control setpoints."""
    return {
        "compressor_loads": [
            COMP_LOADS[action_array[0]],
            COMP_LOADS[action_array[1]],
            STANDBY_LOADS[action_array[2]],
        ],
        "n_hp_pumps":       action_array[3] + 1,   # 1–4
        "n_vaporizers":     action_array[4] + 1,   # 1–4
        "send_out_rate_t_h": send_out_demand,       # demand is given, not chosen
    }
```

### 4.3 Reward function (`src/environment/reward.py`)

```python
import numpy as np


def compute_reward(
    state,
    new_state,
    info: dict,
    config: dict,
    penalty_weights: dict,
) -> tuple[float, dict]:
    """
    Reward = -electricity_cost - constraint_penalties - wear_costs

    Primary objective: minimize electricity cost
    Constraints (with penalties, not hard boundaries):
        - Tank pressure must stay within safe range
        - Send-out demand must be met
        - Minimum inventory must be maintained
        - BOG flaring must be avoided (BOG should be consumed)
    Wear cost:
        - Penalty per compressor start (cycling damage)

    Returns: (reward, info_dict with component breakdown)
    """
    # Primary: electricity cost (negative — we minimize cost)
    cost_eur_h = info["cost_eur_h"]
    reward = -cost_eur_h

    penalties = {}

    # Pressure constraint
    pressure = new_state.tank_pressure_kPa
    max_p = config["tanks"]["max_pressure_kPa"]
    min_p = config["tanks"]["min_pressure_kPa"]
    if pressure > max_p:
        pen = penalty_weights["pressure"] * (pressure - max_p) ** 2
        reward -= pen
        penalties["overpressure"] = pen
    elif pressure < min_p:
        pen = penalty_weights["pressure"] * (min_p - pressure) ** 2
        reward -= pen
        penalties["underpressure"] = pen

    # Inventory constraint (don't run dry, don't overfill)
    fill = new_state.fill_fraction
    if fill < 0.12:  # approaching minimum
        pen = penalty_weights["low_inventory"] * (0.12 - fill) * 1000
        reward -= pen
        penalties["low_inventory"] = pen
    elif fill > 0.98:  # nearly full
        pen = penalty_weights["high_inventory"] * (fill - 0.98) * 1000
        reward -= pen
        penalties["high_inventory"] = pen

    # BOG flaring penalty (if compressors can't keep up with generation)
    bog_gen     = info["bog_gen_kg_h"]
    bog_removed = info["bog_removed_kg_h"]
    bog_excess  = max(0, bog_gen - bog_removed)
    if bog_excess > 50:  # kg/h excess that would need flaring
        pen = penalty_weights["bog_flaring"] * bog_excess
        reward -= pen
        penalties["bog_flaring"] = pen

    # Compressor wear (starts penalty)
    n_new_starts = (
        new_state.compressor_starts_today - state.compressor_starts_today
    )
    if n_new_starts > 0:
        pen = penalty_weights["comp_start"] * n_new_starts
        reward -= pen
        penalties["compressor_wear"] = pen

    # Carbon cost (EU ETS ~€60-80/tCO2)
    # LNG terminal uses electricity — CO2 from grid
    # Lithuanian grid emission factor ≈ 0.3 tCO2/MWh (2024)
    grid_emission_factor = 0.30  # tCO2/MWh
    carbon_price_eur_t = 65.0    # EUR/tCO2 (approximate ETS price)
    energy_mwh = info["total_power_kW"] / 1000
    carbon_cost = energy_mwh * grid_emission_factor * carbon_price_eur_t
    reward -= carbon_cost
    penalties["carbon_cost"] = carbon_cost

    info_out = {
        "reward_total":   reward,
        "electricity_cost": cost_eur_h,
        "carbon_cost":    carbon_cost,
        **penalties,
    }

    return float(reward), info_out
```

### 4.4 Main Gymnasium environment (`src/environment/lng_terminal_env.py`)

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.physics.terminal_simulator import LNGTerminalSimulator, TerminalState
from src.surrogate.pinn import TerminalPINN
from src.market.nordpool_fetcher import load_and_clean_prices
from .state_space import encode_state, STATE_DIM
from .action_space import ACTION_DIMS, decode_action
from .reward import compute_reward
import yaml, torch
from pathlib import Path


class LNGTerminalEnv(gym.Env):
    """
    Gymnasium environment for LNG terminal energy optimization.

    Timestep: 1 hour
    Episode: 1 year (8760 steps) or configurable
    Observation: 44-dimensional state vector
    Action: MultiDiscrete — compressor loads + pump/vaporizer counts
    Reward: -electricity_cost - penalties - wear_cost - carbon_cost

    The environment uses the PINN surrogate for fast simulation.
    The physics simulator is available for validation.

    Key novelty vs existing work:
    - Electricity price in state + 24h forecast enables temporal arbitrage
    - LNG composition evolves dynamically (aging effect)
    - FSRU motion (sea_state) modeled as stochastic disturbance
    - Carbon cost included in reward
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str = "config/terminal.yaml",
        price_data_path: str = "data/nordpool/raw",
        surrogate_path: str = "runs/surrogate/best_pinn.pt",
        demand_profile: str = "flat",  # 'flat', 'seasonal', 'historical'
        use_surrogate: bool = True,
        episode_length_h: int = 8760,
        seed: int = 42,
    ):
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load electricity prices
        self.prices_df = load_and_clean_prices(Path(price_data_path))

        # Load PINN surrogate
        self.use_surrogate = use_surrogate
        if use_surrogate:
            checkpoint = torch.load(surrogate_path, map_location="cpu")
            self.surrogate = TerminalPINN()
            self.surrogate.load_state_dict(checkpoint["model_state"])
            self.surrogate.eval()
        else:
            self.physics_sim = LNGTerminalSimulator(self.config)

        self.episode_length = episode_length_h
        self.rng = np.random.default_rng(seed)

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0,
            shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

        # Penalty weights (tunable)
        self.penalty_weights = {
            "pressure":       100.0,
            "low_inventory":  500.0,
            "high_inventory": 100.0,
            "bog_flaring":    50.0,
            "comp_start":     self.config["bog_compressors"]["wear_cost_per_start"],
        }

        self.state:  TerminalState = None
        self.hour:   int = 0
        self.episode_cost: float = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start within the price dataset
        start_idx = self.rng.integers(
            168, len(self.prices_df) - self.episode_length - 24
        )
        self.price_start_idx = start_idx
        self.hour = 0

        # Random initial state (realistic range)
        self.state = TerminalState(
            liquid_volume_m3   = float(self.rng.uniform(40000, 130000)),
            tank_pressure_kPa  = float(self.rng.uniform(4, 15)),
            lng_temp_K         = float(self.rng.uniform(111, 114)),
            x_methane          = float(self.rng.uniform(0.85, 0.93)),
            x_ethane           = float(self.rng.uniform(0.04, 0.10)),
            x_propane          = float(self.rng.uniform(0.01, 0.04)),
            x_nitrogen         = 0.01,
            ambient_temp_C     = self._get_ambient_temp(),
            seawater_temp_C    = self._get_seawater_temp(),
            sea_state          = float(self.rng.uniform(0, 0.5)),
            electricity_price_eur_mwh = self._get_price(0),
        )
        # Normalize composition
        tot = (self.state.x_methane + self.state.x_ethane +
               self.state.x_propane + self.state.x_nitrogen)
        self.state.x_methane /= tot
        self.state.x_ethane  /= tot
        self.state.x_propane /= tot
        self.state.x_nitrogen /= tot

        self.episode_cost = 0.0
        obs = encode_state(self.state, self._get_price_forecast())
        return obs, {}

    def step(self, action):
        send_out_demand = self._get_send_out_demand()
        ctrl = decode_action(list(action), send_out_demand)

        if self.use_surrogate:
            new_state, info = self._surrogate_step(ctrl)
        else:
            new_state, info = self.physics_sim.step(self.state, ctrl)

        reward, reward_info = compute_reward(
            self.state, new_state, info, self.config, self.penalty_weights
        )
        self.episode_cost += info["cost_eur_h"]
        self.hour += 1
        self.state = new_state
        self.state.electricity_price_eur_mwh = self._get_price(self.hour)
        self.state.ambient_temp_C  = self._get_ambient_temp()
        self.state.seawater_temp_C = self._get_seawater_temp()
        self.state.sea_state       = self._update_sea_state()

        obs  = encode_state(self.state, self._get_price_forecast())
        done = self.hour >= self.episode_length
        info.update(reward_info)
        info["episode_cost_eur"] = self.episode_cost

        return obs, reward, done, False, info

    def _get_price(self, hour_offset: int = 0) -> float:
        idx = min(self.price_start_idx + self.hour + hour_offset,
                  len(self.prices_df) - 1)
        return float(self.prices_df.iloc[idx]["price_eur_mwh"])

    def _get_price_forecast(self) -> np.ndarray:
        return np.array([self._get_price(i) for i in range(1, 25)])

    def _get_ambient_temp(self) -> float:
        month = self.prices_df.iloc[self.price_start_idx + self.hour]["month"]
        # Klaipėda climate: cold winters, mild summers
        monthly_temps = [-3, -3, 1, 6, 12, 16, 18, 17, 13, 8, 3, -1]
        base = monthly_temps[int(month) - 1]
        return base + float(self.rng.normal(0, 3))

    def _get_seawater_temp(self) -> float:
        month = self.prices_df.iloc[self.price_start_idx + self.hour]["month"]
        # Baltic Sea at Klaipėda: cold in winter (3°C), warm in summer (19°C)
        monthly_sea = [3, 3, 4, 6, 11, 15, 18, 19, 15, 11, 7, 4]
        base = monthly_sea[int(month) - 1]
        return max(2.0, base + float(self.rng.normal(0, 1.5)))

    def _get_send_out_demand(self) -> float:
        # Seasonal demand: higher in winter
        month = self.prices_df.iloc[self.price_start_idx + self.hour]["month"]
        winter_months = [12, 1, 2, 3]
        if int(month) in winter_months:
            base_demand = self.rng.uniform(180, 350)
        else:
            base_demand = self.rng.uniform(60, 180)
        return float(np.clip(base_demand, 20, 380))

    def _update_sea_state(self) -> float:
        # Autoregressive sea state with persistence
        current = self.state.sea_state
        noise   = self.rng.normal(0, 0.1)
        return float(np.clip(0.9 * current + noise, 0, 1))

    def _surrogate_step(self, action: dict) -> tuple:
        """Use PINN surrogate instead of physics simulator."""
        import copy
        # Build input vector for PINN
        # [fill, pressure, x_me, x_et, amb, sea, state, price,
        #  c0, c1, c2, n_hp, n_vap, send]
        x = np.array([
            self.state.fill_fraction,
            self.state.tank_pressure_kPa / 25.0,
            self.state.x_methane,
            self.state.x_ethane,
            (self.state.ambient_temp_C + 20) / 55.0,
            (self.state.seawater_temp_C - 2) / 20.0,
            self.state.sea_state,
            self.state.electricity_price_eur_mwh / 300.0,
            action["compressor_loads"][0],
            action["compressor_loads"][1],
            action["compressor_loads"][2],
            (action["n_hp_pumps"] - 1) / 3.0,
            (action["n_vaporizers"] - 1) / 3.0,
            action["send_out_rate_t_h"] / 400.0,
        ], dtype=np.float32)

        with torch.no_grad():
            x_t = torch.tensor(x).unsqueeze(0)
            pred = self.surrogate.predict_physical(x_t).squeeze(0).numpy()

        bog_gen, comp_power, total_power, cost, new_pressure, new_fill = pred

        new_state = copy.deepcopy(self.state)
        new_state.tank_pressure_kPa   = float(new_pressure)
        new_state.liquid_volume_m3    = float(new_fill * 155000)
        new_state.compressor_loads    = action["compressor_loads"]
        new_state.n_hp_pumps_running  = action["n_hp_pumps"]
        new_state.n_vaporizers_running = action["n_vaporizers"]
        new_state.total_power_kW      = float(total_power)
        new_state.electricity_cost_eur_h = float(cost)

        # Composition aging (too fast for PINN to learn — apply analytically)
        aging = 0.0002
        new_state.x_methane = max(0.80, self.state.x_methane - aging)
        new_state.x_ethane  = min(0.14, self.state.x_ethane + aging * 0.6)
        new_state.x_propane = min(0.08, self.state.x_propane + aging * 0.4)
        tot = new_state.x_methane + new_state.x_ethane + new_state.x_propane + new_state.x_nitrogen
        new_state.x_methane /= tot; new_state.x_ethane /= tot
        new_state.x_propane /= tot; new_state.x_nitrogen /= tot

        info = {
            "bog_gen_kg_h":    float(bog_gen),
            "bog_removed_kg_h": float(bog_gen * 0.95),  # approximate
            "comp_power_kW":   float(comp_power),
            "total_power_kW":  float(total_power),
            "cost_eur_h":      float(cost),
            "tank_pressure":   float(new_pressure),
            "fill_fraction":   float(new_fill),
        }

        return new_state, info
```

---

## Phase 5 — RL agent training

### 5.1 Training script (`scripts/train_agent.py`)

```python
import click
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback
)
import mlflow
from pathlib import Path


@click.command()
@click.option("--total-steps",  default=5_000_000, show_default=True)
@click.option("--n-envs",       default=8,          show_default=True)
@click.option("--lr",           default=3e-4,       show_default=True)
@click.option("--batch-size",   default=2048,       show_default=True)
@click.option("--output-dir",   default="runs/agent")
@click.option("--surrogate",    default="runs/surrogate/best_pinn.pt")
def train(total_steps, n_envs, lr, batch_size, output_dir, surrogate):
    from src.environment.lng_terminal_env import LNGTerminalEnv

    def make_env():
        return LNGTerminalEnv(
            surrogate_path=surrogate,
            use_surrogate=True,
            episode_length_h=8760,  # full year
        )

    # Vectorized environments for parallel training
    vec_env  = make_vec_env(make_env, n_envs=n_envs)
    eval_env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,            # discount — 1 year horizon needs high gamma
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # entropy bonus for exploration
        verbose=1,
        tensorboard_log=f"{output_dir}/tb",
        policy_kwargs=dict(
            net_arch=[256, 256, 128],
            activation_fn=__import__("torch.nn", fromlist=["Tanh"]).Tanh,
        ),
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=f"{output_dir}/best",
            log_path=f"{output_dir}/eval_logs",
            eval_freq=50_000,
            n_eval_episodes=5,
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=100_000,
            save_path=f"{output_dir}/checkpoints",
        ),
    ]

    with mlflow.start_run(run_name="ppo_lng_agent"):
        mlflow.log_params({
            "total_steps": total_steps,
            "n_envs": n_envs,
            "lr": lr,
            "batch_size": batch_size,
        })
        model.learn(total_timesteps=total_steps, callback=callbacks)
        model.save(f"{output_dir}/final_model")
        print(f"\nTraining complete. Model saved to {output_dir}/final_model")

if __name__ == "__main__":
    train()
```

### 5.2 Baseline agents (`src/agent/baseline_agents.py`)

```python
import numpy as np


class RuleBasedAgent:
    """
    Current industry practice: run fixed number of compressors
    based on tank pressure. No price awareness, no forecasting.
    This is what the RL agent competes against.
    """
    def predict(self, obs: np.ndarray, state=None, episode_start=None) -> tuple:
        pressure = obs[1] * 25.0  # denormalize
        # Simple rule: start/stop based on pressure thresholds
        if pressure > 18:    # high pressure → run 2 compressors full
            action = [3, 3, 0, 2, 2]
        elif pressure > 12:  # normal → run 1 compressor at 60%
            action = [2, 0, 0, 2, 2]
        else:                # low pressure → minimum operation
            action = [1, 0, 0, 1, 1]
        return np.array(action), state


class PriceAwareHeuristic:
    """
    Improved baseline: same pressure-based logic but shifts high loads
    to low-price hours. No RL, just a simple price threshold rule.
    Demonstrates value of price awareness without RL complexity.
    """
    def predict(self, obs: np.ndarray, state=None, episode_start=None) -> tuple:
        pressure    = obs[1] * 25.0
        price       = obs[7] * 300.0          # current price
        future_min  = obs[8:32].min() * 300.0  # cheapest hour in next 24h
        price_high  = price > future_min * 1.5  # current price is expensive

        if pressure > 20:    # must compress regardless of price
            action = [3, 3, 0, 2, 2]
        elif pressure > 15 and not price_high:
            action = [2, 2, 0, 2, 2]
        elif pressure > 10 and not price_high:
            action = [2, 0, 0, 2, 2]
        elif price_high:     # defer compression if price is high
            action = [1, 0, 0, 1, 2]
        else:
            action = [1, 0, 0, 1, 1]

        return np.array(action), state
```

---

## Phase 6 — Evaluation and ablation

### 6.1 Key metrics (`src/eval/metrics.py`)

```python
import numpy as np
import pandas as pd


def annual_cost_savings(
    rl_costs_eur: list[float],
    baseline_costs_eur: list[float],
) -> dict:
    """
    Compare annual electricity costs: RL agent vs rule-based baseline.
    This is the primary result for the paper.
    """
    rl_total   = sum(rl_costs_eur)
    base_total = sum(baseline_costs_eur)
    savings    = base_total - rl_total
    savings_pct = savings / base_total * 100

    return {
        "rl_annual_cost_eur":       rl_total,
        "baseline_annual_cost_eur": base_total,
        "savings_eur":              savings,
        "savings_pct":              savings_pct,
    }


def constraint_violation_rate(episode_infos: list[dict]) -> dict:
    """
    Safety metric: what fraction of timesteps violated each constraint?
    Must be near zero for a deployable controller.
    """
    n = len(episode_infos)
    return {
        "overpressure_rate":   sum("overpressure" in i for i in episode_infos) / n,
        "underpressure_rate":  sum("underpressure" in i for i in episode_infos) / n,
        "low_inventory_rate":  sum("low_inventory" in i for i in episode_infos) / n,
        "bog_flaring_rate":    sum("bog_flaring" in i for i in episode_infos) / n,
    }


def price_correlation_analysis(
    actions: np.ndarray,     # (T, 5) action array
    prices: np.ndarray,      # (T,) electricity prices
) -> float:
    """
    Key novelty verification: does the agent actually shift load to
    low-price periods?
    Compute correlation between power consumption and electricity price.
    Good agent: negative correlation (use less power when expensive).
    Rule-based: near-zero correlation.
    """
    comp_load = actions[:, 0] + actions[:, 1]  # total compressor load
    return float(np.corrcoef(comp_load, prices)[0, 1])
```

### 6.2 Ablation study (`src/eval/ablation.py`)

```python
"""
Ablation study: remove each novelty and measure impact on annual cost.
This is essential for the paper — proves each component contributes.

Ablations:
A) No price forecast in state (obs[8:32] zeroed out)
   → Quantifies value of price forecasting
B) No composition tracking (x_methane, x_ethane fixed at nominal)
   → Quantifies value of composition as state variable
C) No FSRU motion term (sea_state always 0)
   → Quantifies value of FSRU-specific modeling
D) No carbon cost in reward
   → Quantifies carbon cost impact on behavior
E) Full model (all novelties included)
   → Best performance

Present as a table: Ablation | Annual Cost | Savings vs Baseline | Delta vs Full
"""
```

---

## Tests

```python
# tests/test_thermodynamics.py
def test_bog_rate_physical_range():
    from src.physics.thermodynamics import LNGComposition, compute_bog_rate_physics
    comp = LNGComposition()
    bog = compute_bog_rate_physics(180.0, comp)
    assert 100 < bog < 400, f"BOG rate {bog:.1f} kg/h outside physical range"

# tests/test_bog_model.py
def test_sea_state_increases_bog():
    from src.physics.bog_model import BOGModel
    import yaml
    config = yaml.safe_load(open("config/terminal.yaml"))
    model = BOGModel(config)
    comp = __import__("src.physics.thermodynamics", fromlist=["LNGComposition"]).LNGComposition()
    calm = model.steady_state_bog(comp, 0.7, 10.0, sea_state=0.0)
    rough = model.steady_state_bog(comp, 0.7, 10.0, sea_state=1.0)
    assert rough > calm, "Rough sea should produce more BOG"

# tests/test_environment.py
def test_env_reset_and_step():
    from src.environment.lng_terminal_env import LNGTerminalEnv
    env = LNGTerminalEnv(use_surrogate=False)
    obs, _ = env.reset()
    assert obs.shape == (44,)
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs2, reward, done, _, info = env.step(action)
    assert obs2.shape == (44,)
    assert isinstance(reward, float)
    assert "cost_eur_h" in info

# tests/test_reward.py
def test_high_price_penalizes_more():
    """Agent should learn to use less power when prices are high."""
    # Create two scenarios: same action, high vs low price
    # High price should produce lower (more negative) reward
    pass  # implement based on reward function signature
```

---

## CLAUDE.md (paste at start of every Claude Code session)

```markdown
# LNG Terminal RL Energy Optimizer — Session Context

## What this is
Safe RL agent that minimizes electricity cost for LNG terminal operations.
Novel vs existing work (RALT-DT 2025) in three ways:
1. Optimizes electricity COST vs NordPool prices, not just energy volume
2. LNG composition (methane/ethane/propane) is a dynamic state variable
3. Physics-informed neural network surrogate instead of black-box NN

Target: Klaipėda FSRU Independence (publicly documented 170,000 m³ FSRU).

## Stack
Python 3.11 | CoolProp | PyTorch | Gymnasium | Stable-Baselines3 | MLflow

## Pipeline
NordPool prices → price_forecaster.py
Physics simulator → surrogate training data → TerminalPINN
TerminalPINN + price forecaster → LNGTerminalEnv (Gymnasium)
LNGTerminalEnv → PPO agent (SB3) → annual cost savings vs rule-based baseline

## State vector (44 dims)
[0]   fill_fraction          [1]   tank_pressure_norm
[2]   x_methane              [3]   x_ethane
[4]   ambient_temp_norm      [5]   seawater_temp_norm
[6]   sea_state              [7]   current_price_norm
[8:32] price_forecast_24h   [32:34] hour sin/cos
[35:37] compressor_loads    [38:39] n_pumps, n_vap
[40:43] wear tracking

## Action space (MultiDiscrete [4,4,3,4,4])
comp_0 load, comp_1 load, comp_2 load, n_hp_pumps, n_vaporizers

## Reward
-electricity_cost_eur_h - carbon_cost - pressure_penalties - bog_penalties - wear_cost

## Key source files
src/physics/terminal_simulator.py  — ODE physics model (surrogate training)
src/physics/bog_model.py           — BOG generation (FSRU-aware)
src/surrogate/pinn.py              — TerminalPINN with physics constraints
src/market/nordpool_fetcher.py     — NordPool Baltic price data
src/environment/lng_terminal_env.py — Gymnasium env (core)
src/environment/reward.py          — cost + penalties + carbon
src/agent/baseline_agents.py       — rule-based + price-heuristic baselines

## Paper contribution framing
Primary result: annual electricity cost savings (EUR) vs rule-based baseline.
Secondary results: constraint violation rate, price-load correlation (novelty proof).
Ablation: show each of 3 novelties independently contributes to savings.

## Current status
[Update at end of each session]

## Next task
[Write specifically what to implement next]
```

---

## First 3 commands after setup

```bash
# 1. Check environment
python scripts/check_env.py

# 2. Validate physics model (before anything ML)
python -c "
from src.physics.terminal_simulator import LNGTerminalSimulator, TerminalState
import yaml
config = yaml.safe_load(open('config/terminal.yaml'))
sim = LNGTerminalSimulator(config)
state = TerminalState(liquid_volume_m3=100000, tank_pressure_kPa=10,
                      lng_temp_K=112)
action = {'compressor_loads': [0.8, 0.6, 0.0], 'n_hp_pumps': 2,
          'n_vaporizers': 2, 'send_out_rate_t_h': 150}
new_state, info = sim.step(state, action)
print('BOG gen:', info['bog_gen_kg_h'], 'kg/h')
print('Total power:', info['total_power_kW'], 'kW')
print('Cost:', info['cost_eur_h'], 'EUR/h')
assert 100 < info['bog_gen_kg_h'] < 400, 'BOG rate unphysical'
assert 500 < info['total_power_kW'] < 5000, 'Power unphysical'
print('Physics model OK')
"

# 3. Download price data (requires ENTSOE_TOKEN env var)
python scripts/download_nordpool_data.py
```

If the physics validation step produces unphysical values, fix the
simulator before proceeding — everything downstream depends on it.
```
