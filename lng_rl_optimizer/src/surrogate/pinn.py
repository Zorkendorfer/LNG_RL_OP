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

    Outputs (5):
        bog_gen_kg_h, comp_power_kW, total_power_kW,
        new_pressure, new_fill
    """

    def __init__(
        self,
        input_dim:  int = 14,
        hidden_dim: int = 256,
        n_layers:   int = 6,
        output_dim: int = 5,
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

        self.register_buffer("out_mean", torch.tensor([
            200.0, 1500.0, 2500.0, 10.0, 0.5,
        ]))
        self.register_buffer("out_std", torch.tensor([
            150.0, 800.0, 1200.0, 5.0, 0.2
        ]))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))

    def predict_physical(self, x: Tensor) -> Tensor:
        raw = self.forward(x)
        physical = raw * self.out_std + self.out_mean
        physical = torch.clamp(physical, min=torch.zeros(5, device=x.device))
        return physical


def physics_consistency_loss(
    pred: Tensor,
    inputs: Tensor,
    weight: float = 1.0,
) -> Tensor:
    """Soft physics constraints as additional loss terms."""
    bog_gen     = pred[:, 0]
    comp_power  = pred[:, 1]
    total_power = pred[:, 2]
    price       = inputs[:, 7]
    send_out    = inputs[:, 13] * 400

    c1 = torch.relu(comp_power - total_power)
    cost_physics = total_power * price / 1000
    c2 = torch.relu(cost_physics - total_power * 0.6)
    c3 = torch.relu(-bog_gen)
    c4 = torch.relu(-comp_power)
    min_power_when_running = 200.0
    c5 = torch.relu(
        (send_out > 10).float() * (min_power_when_running - total_power)
    )

    return weight * (c1.mean() + c2.mean() + c3.mean() + c4.mean() + c5.mean())
