import numpy as np

# Discrete action space
COMP_LOADS    = [0.0, 0.3, 0.6, 1.0]
STANDBY_LOADS = [0.0, 0.3, 0.6]
ACTION_DIMS   = [4, 4, 3, 4, 4]   # MultiDiscrete shape


def decode_action(action_array: list, send_out_demand: float) -> dict:
    """Convert discrete action indices to physical control setpoints."""
    return {
        "compressor_loads": [
            COMP_LOADS[action_array[0]],
            COMP_LOADS[action_array[1]],
            STANDBY_LOADS[action_array[2]],
        ],
        "n_hp_pumps":        action_array[3] + 1,
        "n_vaporizers":      action_array[4] + 1,
        "send_out_rate_t_h": send_out_demand,
    }
