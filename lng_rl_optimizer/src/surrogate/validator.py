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
    Returns per-output validation metrics.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    x_mean = torch.tensor(checkpoint["x_mean"], dtype=torch.float32)
    x_std  = torch.tensor(checkpoint["x_std"],  dtype=torch.float32)

    dfs = [pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))]
    df_all = pd.concat(dfs, ignore_index=True)
    df = df_all.sample(n=min(n_samples, len(df_all)), random_state=42)

    X_raw = torch.tensor(df[INPUT_COLS].values, dtype=torch.float32)
    X = (X_raw - x_mean) / x_std
    Y = torch.tensor(df[OUTPUT_COLS].values, dtype=torch.float32)
    price = torch.tensor(df["price_eur_mwh"].values, dtype=torch.float32)
    cost_true = torch.tensor(df["cost_eur_h"].values, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred = model.predict_physical(X)
        cost_pred = pred[:, 2] * price / 1000.0

    errors = {}
    for i, col in enumerate(OUTPUT_COLS):
        abs_err = (pred[:, i] - Y[:, i]).abs()
        mae = abs_err.mean().item()
        rmse = torch.sqrt(((pred[:, i] - Y[:, i]) ** 2).mean()).item()
        scale = torch.maximum(Y[:, i].abs(), torch.full_like(Y[:, i], 1.0))
        rel_err = (abs_err / scale).mean().item()
        errors[col] = {"mae": mae, "rmse": rmse, "rel_err": rel_err}
        print(
            f"{col:25s}: mae={mae:.3f} | rmse={rmse:.3f} | "
            f"rel_err={rel_err:.3f} ({rel_err*100:.1f}%)"
        )

    cost_abs_err = (cost_pred - cost_true).abs()
    cost_mae = cost_abs_err.mean().item()
    cost_rmse = torch.sqrt(((cost_pred - cost_true) ** 2).mean()).item()
    cost_scale = torch.maximum(cost_true.abs(), torch.full_like(cost_true, 1.0))
    cost_rel_err = (cost_abs_err / cost_scale).mean().item()
    errors["cost_eur_h"] = {
        "mae": cost_mae,
        "rmse": cost_rmse,
        "rel_err": cost_rel_err,
    }
    print(
        f"{'cost_eur_h':25s}: mae={cost_mae:.3f} | rmse={cost_rmse:.3f} | "
        f"rel_err={cost_rel_err:.3f} ({cost_rel_err*100:.1f}%)"
    )

    return errors
