import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pathlib import Path
import yaml
import torch

from src.physics.terminal_simulator import LNGTerminalSimulator, TerminalState
from src.surrogate.pinn import TerminalPINN
from .state_space import encode_state, STATE_DIM
from .action_space import ACTION_DIMS, decode_action
from .reward import compute_reward


class LNGTerminalEnv(gym.Env):
    """
    Gymnasium environment for LNG terminal energy optimization.

    Timestep: 1 hour
    Episode:  1 year (8760 steps) or configurable
    Observation: 44-dimensional state vector
    Action: MultiDiscrete — compressor loads + pump/vaporizer counts
    Reward: -electricity_cost - penalties - wear_cost - carbon_cost
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path:      str = "config/terminal.yaml",
        price_data_path:  str = "data/nordpool/raw",
        surrogate_path:   str = "runs/surrogate/best_pinn.pt",
        use_surrogate:    bool = True,
        episode_length_h: int = 8760,
        seed:             int = 42,
        use_synthetic_prices: bool = False,
    ):
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load electricity prices
        if use_synthetic_prices:
            from src.market.nordpool_fetcher import generate_synthetic_prices
            self.prices_df = generate_synthetic_prices()
        else:
            from src.market.nordpool_fetcher import load_and_clean_prices
            self.prices_df = load_and_clean_prices(Path(price_data_path))

        # Load PINN surrogate or physics sim
        self.use_surrogate = use_surrogate
        if use_surrogate and Path(surrogate_path).exists():
            checkpoint = torch.load(
                surrogate_path, map_location="cpu", weights_only=False
            )
            self.surrogate = TerminalPINN()
            self.surrogate.load_state_dict(checkpoint["model_state"])
            self.surrogate.eval()
            self._x_mean = torch.tensor(checkpoint["x_mean"], dtype=torch.float32)
            self._x_std  = torch.tensor(checkpoint["x_std"],  dtype=torch.float32)
        else:
            self.use_surrogate = False
            self.physics_sim = LNGTerminalSimulator(self.config)

        self.episode_length = episode_length_h
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(STATE_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(ACTION_DIMS)

        self.penalty_weights = {
            "pressure":       100.0,
            "low_inventory":  500.0,
            "high_inventory": 100.0,
            "bog_flaring":     50.0,
            "comp_start":     self.config["bog_compressors"]["wear_cost_per_start"],
        }

        self.state:        TerminalState = None
        self.hour:         int = 0
        self.episode_cost: float = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        start_idx = self.rng.integers(
            168, max(169, len(self.prices_df) - self.episode_length - 24)
        )
        self.price_start_idx = int(start_idx)
        self.hour = 0

        x_me, x_et, x_pr, x_ni = (
            float(self.rng.uniform(0.85, 0.93)),
            float(self.rng.uniform(0.04, 0.10)),
            float(self.rng.uniform(0.01, 0.04)),
            0.01,
        )
        tot = x_me + x_et + x_pr + x_ni

        self.state = TerminalState(
            liquid_volume_m3=float(self.rng.uniform(40000, 130000)),
            tank_pressure_kPa=float(self.rng.uniform(4, 15)),
            lng_temp_K=float(self.rng.uniform(111, 114)),
            x_methane=x_me / tot,
            x_ethane=x_et / tot,
            x_propane=x_pr / tot,
            x_nitrogen=x_ni / tot,
            ambient_temp_C=self._get_ambient_temp(),
            seawater_temp_C=self._get_seawater_temp(),
            sea_state=float(self.rng.uniform(0, 0.5)),
            electricity_price_eur_mwh=self._get_price(0),
        )
        self.episode_cost = 0.0
        obs = encode_state(
            self.state, self._get_price_forecast(),
            hour=self.hour % 24, day_of_week=(self.hour // 24) % 7
        )
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
        self.state.ambient_temp_C            = self._get_ambient_temp()
        self.state.seawater_temp_C           = self._get_seawater_temp()
        self.state.sea_state                 = self._update_sea_state()

        obs  = encode_state(
            self.state, self._get_price_forecast(),
            hour=self.hour % 24, day_of_week=(self.hour // 24) % 7
        )
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
        idx   = min(self.price_start_idx + self.hour, len(self.prices_df) - 1)
        month = int(self.prices_df.iloc[idx]["month"])
        monthly_temps = [-3, -3, 1, 6, 12, 16, 18, 17, 13, 8, 3, -1]
        base  = monthly_temps[month - 1]
        return base + float(self.rng.normal(0, 3))

    def _get_seawater_temp(self) -> float:
        idx   = min(self.price_start_idx + self.hour, len(self.prices_df) - 1)
        month = int(self.prices_df.iloc[idx]["month"])
        monthly_sea = [3, 3, 4, 6, 11, 15, 18, 19, 15, 11, 7, 4]
        base  = monthly_sea[month - 1]
        return max(2.0, base + float(self.rng.normal(0, 1.5)))

    def _get_send_out_demand(self) -> float:
        idx   = min(self.price_start_idx + self.hour, len(self.prices_df) - 1)
        month = int(self.prices_df.iloc[idx]["month"])
        if month in [12, 1, 2, 3]:
            base = float(self.rng.uniform(180, 350))
        else:
            base = float(self.rng.uniform(60, 180))
        return float(np.clip(base, 20, 380))

    def _update_sea_state(self) -> float:
        noise = self.rng.normal(0, 0.1)
        return float(np.clip(0.9 * self.state.sea_state + noise, 0, 1))

    def _surrogate_step(self, action: dict) -> tuple:
        import copy
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

        # Apply input normalization from training
        if hasattr(self, "_x_mean"):
            x_t = (torch.tensor(x) - self._x_mean) / self._x_std
        else:
            x_t = torch.tensor(x)

        with torch.no_grad():
            pred = self.surrogate.predict_physical(x_t.unsqueeze(0)).squeeze(0).numpy()

        bog_gen, comp_power, total_power, new_pressure, new_fill = pred
        cost = float(total_power) * self.state.electricity_price_eur_mwh / 1000.0

        new_state = copy.deepcopy(self.state)
        new_state.tank_pressure_kPa    = float(np.clip(new_pressure, 0, 25))
        new_state.liquid_volume_m3     = float(np.clip(new_fill * 155000, 17000, 155000))
        new_state.compressor_loads     = action["compressor_loads"]
        new_state.n_hp_pumps_running   = action["n_hp_pumps"]
        new_state.n_vaporizers_running = action["n_vaporizers"]
        new_state.total_power_kW       = float(total_power)
        new_state.electricity_cost_eur_h = float(cost)

        aging = 0.0002
        new_state.x_methane = max(0.80, self.state.x_methane - aging)
        new_state.x_ethane  = min(0.14, self.state.x_ethane  + aging * 0.6)
        new_state.x_propane = min(0.08, self.state.x_propane + aging * 0.4)
        tot = (new_state.x_methane + new_state.x_ethane +
               new_state.x_propane + new_state.x_nitrogen)
        new_state.x_methane  /= tot
        new_state.x_ethane   /= tot
        new_state.x_propane  /= tot
        new_state.x_nitrogen /= tot

        info = {
            "bog_gen_kg_h":    float(bog_gen),
            "bog_removed_kg_h": float(bog_gen * 0.95),
            "comp_power_kW":   float(comp_power),
            "total_power_kW":  float(total_power),
            "cost_eur_h":      float(cost),
            "tank_pressure":   float(new_pressure),
            "fill_fraction":   float(new_fill),
        }
        return new_state, info
