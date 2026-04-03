import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .pinn import TerminalPINN
from .trainer import INPUT_COLS, OUTPUT_COLS


def validate_surrogate(
    model: TerminalPINN,
    data_dir: Path,
    checkpoint_path: Path,
    n_samples: int = 5000,
) -> dict:
    """
    Compare PINN predictions vs physics simulator outputs on held-out data.
    Returns per-output relative errors.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    x_mean = torch.tensor(checkpoint["x_mean"], dtype=torch.float32)
    x_std  = torch.tensor(checkpoint["x_std"],  dtype=torch.float32)

    dfs = [pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))]
    df  = pd.concat(dfs, ignore_index=True).sample(n=min(n_samples, len(pd.concat(dfs))))

    X_raw = torch.tensor(df[INPUT_COLS].values, dtype=torch.float32)
    X = (X_raw - x_mean) / x_std
    Y = torch.tensor(df[OUTPUT_COLS].values, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model.predict_physical(X)

    errors = {}
    for i, col in enumerate(OUTPUT_COLS):
        rel_err = ((pred[:, i] - Y[:, i]).abs() / (Y[:, i].abs() + 1e-8)).mean().item()
        errors[col] = rel_err
        print(f"{col:25s}: mean relative error = {rel_err:.3f} ({rel_err*100:.1f}%)")

    return errors
