import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class PriceForecaster(nn.Module):
    """
    LSTM-based 24-hour ahead electricity price forecaster.
    Input:  last 168 hours (1 week) of prices + calendar features
    Output: 24-hour price forecast [EUR/MWh]
    """

    def __init__(
        self,
        input_dim:  int = 6,
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
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def prepare_price_features(
    prices_df: pd.DataFrame,
    lookback_h: int = 168,
) -> tuple:
    """Prepare (X, y) pairs for price forecaster training."""
    price     = prices_df["price_eur_mwh"].values
    price_log = np.log1p(np.clip(price, 0, None))
    hour      = prices_df["hour"].values / 23.0
    dow       = prices_df["day_of_week"].values / 6.0
    month     = (prices_df["month"].values - 1) / 11.0
    weekend   = prices_df["is_weekend"].values.astype(float)
    price_norm = np.clip(price / 300.0, 0, 1)

    features = np.stack(
        [price_norm, price_log / 6, hour, dow, month, weekend], axis=1
    )

    X, y = [], []
    for i in range(lookback_h, len(price) - 24):
        X.append(features[i - lookback_h:i])
        y.append(price[i:i + 24] / 300.0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_forecaster(
    prices_df: pd.DataFrame,
    output_path: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> PriceForecaster:
    """Train and save the price forecaster."""
    from torch.utils.data import DataLoader, TensorDataset

    X, y = prepare_price_features(prices_df)
    n = len(X)
    n_train = int(n * 0.85)
    X_tr = torch.tensor(X[:n_train])
    y_tr = torch.tensor(y[:n_train])
    X_val = torch.tensor(X[n_train:])
    y_val = torch.tensor(y[n_train:])

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    model = PriceForecaster().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(
                model(X_val.to(device)), y_val.to(device)
            ).item()
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_path)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} val_loss={val_loss:.4f}")

    return model
