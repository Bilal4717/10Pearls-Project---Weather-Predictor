"""Model training utilities: LSTM dataset, metrics, reproducibility."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

import config

logger = logging.getLogger(__name__)


def set_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for LSTM sequences over multivariate time series."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: int = config.LSTM_SEQUENCE_LENGTH,
    ) -> None:
        """Initialize dataset.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target matrix (n_samples, n_targets).
            sequence_length: Lookback window length.
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, len(self.X) - self.sequence_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.sequence_length
        x_seq = self.X[start:end]
        y_out = self.y[end - 1]
        return torch.tensor(x_seq), torch.tensor(y_out)


class LSTMRegressor(torch.nn.Module):
    """Two-layer LSTM for multi-output AQI forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_targets: int = 3,
        dropout: float = 0.2,
    ) -> None:
        """Initialize LSTM architecture.

        Args:
            input_size: Number of input features.
            hidden_size: LSTM hidden dimension.
            num_targets: Number of output targets.
            dropout: Dropout rate between LSTM layers.
        """
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = torch.nn.Linear(hidden_size, num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def time_based_split(
    df,
    feature_cols: List[str],
) -> Tuple[np.ndarray, ...]:
    """Split data chronologically into train/val/test.

    Args:
        df: Sorted feature DataFrame.
        feature_cols: Feature column names.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test arrays.
    """
    n = len(df)
    train_end = int(n * config.TRAIN_RATIO)
    val_end = int(n * (config.TRAIN_RATIO + config.VAL_RATIO))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    X_train = train[feature_cols].values
    X_val = val[feature_cols].values
    X_test = test[feature_cols].values
    y_train = train[config.TARGET_COLUMNS].values
    y_val = val[config.TARGET_COLUMNS].values
    y_test = test[config.TARGET_COLUMNS].values
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Compute RMSE, MAE, R² per target horizon.

    Args:
        y_true: Ground truth (n, 3).
        y_pred: Predictions (n, 3).

    Returns:
        Nested dict of metrics per target.
    """
    metrics = {}
    for i, col in enumerate(config.TARGET_COLUMNS):
        rmse = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        mae = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        r2 = float(r2_score(y_true[:, i], y_pred[:, i]))
        metrics[col] = {"rmse": rmse, "mae": mae, "r2": r2}
    avg_rmse = float(np.mean([metrics[c]["rmse"] for c in config.TARGET_COLUMNS]))
    metrics["avg_rmse"] = avg_rmse
    return metrics


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> LSTMRegressor:
    """Train LSTM regressor with early stopping on validation loss.

    Args:
        X_train: Scaled training features.
        y_train: Training targets.
        X_val: Scaled validation features.
        y_val: Validation targets.
        epochs: Max training epochs.
        batch_size: Batch size.
        lr: Learning rate.

    Returns:
        Trained LSTM model in eval mode.
    """
    set_seeds()
    seq_len = config.LSTM_SEQUENCE_LENGTH
    model = LSTMRegressor(input_size=X_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TimeSeriesDataset(X_train, y_train, seq_len)
    val_ds = TimeSeriesDataset(X_val, y_val, seq_len)
    if len(train_ds) == 0:
        raise ValueError("Insufficient training samples for LSTM sequence length.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    stale = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_losses.append(criterion(pred, yb).item())
        val_loss = np.mean(val_losses) if val_losses else float("inf")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_lstm(model: LSTMRegressor, X: np.ndarray) -> np.ndarray:
    """Generate LSTM predictions for all valid sequence positions.

    Args:
        model: Trained LSTM.
        X: Scaled feature matrix.

    Returns:
        Predictions aligned to sequence endpoints.
    """
    seq_len = config.LSTM_SEQUENCE_LENGTH
    ds = TimeSeriesDataset(X, np.zeros((len(X), len(config.TARGET_COLUMNS))), seq_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            preds.append(model(xb).numpy())
    if not preds:
        return np.zeros((0, len(config.TARGET_COLUMNS)))
    return np.vstack(preds)
