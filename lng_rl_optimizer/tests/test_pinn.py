import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.surrogate.pinn import TerminalPINN, physics_consistency_loss


def test_pinn_output_shape():
    model = TerminalPINN()
    x = torch.randn(8, 14)
    out = model(x)
    assert out.shape == (8, 6)


def test_pinn_predict_physical_nonnegative():
    model = TerminalPINN()
    x = torch.randn(16, 14)
    pred = model.predict_physical(x)
    assert (pred >= 0).all(), "Physical predictions must be non-negative"


def test_physics_loss_nonnegative():
    model = TerminalPINN()
    x = torch.randn(32, 14)
    x[:, 7] = torch.rand(32) * 300  # price column
    x[:, 13] = torch.rand(32)       # send-out column
    pred = model.predict_physical(x)
    loss = physics_consistency_loss(pred, x, weight=1.0)
    assert loss.item() >= 0


def test_pinn_deterministic():
    model = TerminalPINN()
    model.eval()
    x = torch.randn(4, 14)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.allclose(out1, out2)
