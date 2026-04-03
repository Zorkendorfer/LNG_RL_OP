import torch
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pandas as pd
import numpy as np
from .pinn import TerminalPINN, physics_consistency_loss


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


def train_surrogate(
    data_dir: Path,
    output_dir: Path,
    hidden_dim: int = 256,
    n_layers: int = 6,
    lr: float = 3e-4,
    epochs: int = 200,
    batch_size: int = 1024,
    physics_weight: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> TerminalPINN:

    dfs = [pd.read_parquet(p) for p in sorted(data_dir.glob("*.parquet"))]
    df  = pd.concat(dfs, ignore_index=True)

    x_mean = df[INPUT_COLS].mean()
    x_std  = df[INPUT_COLS].std().clip(lower=1e-6)
    X = torch.tensor(
        ((df[INPUT_COLS] - x_mean) / x_std).values, dtype=torch.float32
    )
    Y = torch.tensor(df[OUTPUT_COLS].values, dtype=torch.float32)

    n = len(X)
    n_train = int(n * 0.85)
    idx = torch.randperm(n)
    X_train, Y_train = X[idx[:n_train]], Y[idx[:n_train]]
    X_val,   Y_val   = X[idx[n_train:]], Y[idx[n_train:]]

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size, shuffle=True
    )

    model = TerminalPINN(hidden_dim=hidden_dim, n_layers=n_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Y_val_norm = (Y_val - model.out_mean.cpu()) / model.out_std.cpu()

    best_val_loss = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="pinn_surrogate"):
        mlflow.log_params({
            "hidden_dim": hidden_dim, "n_layers": n_layers,
            "physics_weight": physics_weight, "n_train": n_train,
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

            model.eval()
            with torch.no_grad():
                X_v     = X_val.to(device)
                val_pred = model(X_v)
                val_loss = nn.functional.mse_loss(
                    val_pred, Y_val_norm.to(device)
                ).item()
                pred_phys = model.predict_physical(X_v).cpu()
                rel_l2 = (
                    torch.norm(pred_phys - Y_val) / (torch.norm(Y_val) + 1e-8)
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
