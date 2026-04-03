import numpy as np


class SafetyLayer:
    """
    Safety constraint projection for the RL agent.
    Projects proposed actions into the feasible set defined by:
    - Tank pressure limits
    - Minimum compressor operation (can't run below min_load)
    - Maximum send-out capacity
    """

    def __init__(self, config: dict):
        self.config = config

    def is_safe(self, state, action: dict) -> bool:
        """Check if action is safe given current state."""
        pressure = state.tank_pressure_kPa
        max_p    = self.config["tanks"]["max_pressure_kPa"]
        min_p    = self.config["tanks"]["min_pressure_kPa"]

        # If pressure is high, must have at least one compressor running
        if pressure > max_p * 0.9:
            total_comp_load = sum(action["compressor_loads"])
            if total_comp_load < 0.3:
                return False

        # If fill is critically low, must reduce send-out
        fill = state.liquid_volume_m3 / 155000.0
        if fill < 0.11 and action["send_out_rate_t_h"] > 50:
            return False

        return True

    def project_action(self, state, action: dict) -> dict:
        """Project action to nearest safe action."""
        import copy
        safe_action = copy.deepcopy(action)
        pressure = state.tank_pressure_kPa
        max_p    = self.config["tanks"]["max_pressure_kPa"]

        if pressure > max_p * 0.9:
            if sum(safe_action["compressor_loads"]) < 0.3:
                safe_action["compressor_loads"][0] = 0.6

        fill = state.liquid_volume_m3 / 155000.0
        if fill < 0.11:
            safe_action["send_out_rate_t_h"] = min(
                safe_action["send_out_rate_t_h"], 50.0
            )

        return safe_action
