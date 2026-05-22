"""
LSTM model for short-term volatility forecasting.

Sequences are built per stock with sliding windows including the current
timestep. One shared LSTM is trained on all stocks using strict temporal
split. Output matches the backend runner format for seamless integration
with Erik's ModelSpec pipeline.

Usage within config:
    ModelSpec(name="LSTM", model_type="LSTM", parameters={})

Output saved to artifacts/ in the standard runner format.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise ImportError("PyTorch not found. Install with: pip install torch")

from .config import ModelSpec

# ── Hyperparameters ────────────────────────────────────────────────────────────

SEQ_LEN = 20
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 5e-4
BATCH_SIZE = 256
MAX_EPOCHS = 100
PATIENCE = 10
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ──────────────────────────────────────────────────────────────────────

class LSTMVolatilityModel(nn.Module):
    """
    Stacked LSTM with two-layer MLP head and layer normalisation.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.layer_norm(out[:, -1, :])
        return self.head(out).squeeze(-1)


# ── Sequence construction ──────────────────────────────────────────────────────

def make_sequences(
    data: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str],
    cutoff_time_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding window sequences per stock.

    Each sequence spans [i - seq_len + 1 ... i] (inclusive). Train/test split
    based on whether the LAST time_id in each sequence <= cutoff.

    Returns: X_train, y_train, X_test, y_test, train_indices, test_indices
    """
    X_train, y_train, idx_train = [], [], []
    X_test, y_test, idx_test = [], [], []

    for _, group in data.groupby("stock_id"):
        group = group.sort_values("time_id").reset_index(drop=True)
        X_all = group[feature_cols].values.astype(np.float32)
        y_all = group["target_vol"].values.astype(np.float32)
        tids = group["time_id"].values
        group_idx = group.index.values

        for i in range(seq_len - 1, len(X_all)):
            seq_X = X_all[i - seq_len + 1 : i + 1]
            seq_y = y_all[i]
            idx = group_idx[i]

            if tids[i] <= cutoff_time_id:
                X_train.append(seq_X)
                y_train.append(seq_y)
                idx_train.append(idx)
            else:
                X_test.append(seq_X)
                y_test.append(seq_y)
                idx_test.append(idx)

    return (
        np.array(X_train),
        np.array(y_train),
        np.array(X_test),
        np.array(y_test),
        np.array(idx_train),
        np.array(idx_test),
    )


# ── Training helpers ───────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(y_batch)
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())
    preds = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    return total_loss / len(loader.dataset), preds, actuals


# ── Main runner ────────────────────────────────────────────────────────────────

def run_lstm_model(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    fold: int,
) -> pd.DataFrame:
    """
    Run LSTM model matching Erik's backend runner signature.

    Takes train/test DataFrames (with columns: stock_id, time_id, target_var,
    target_vol, and feature columns), returns predictions in standard format.
    """
    t0 = time.perf_counter()

    # Temporal split within this fold (find cutoff time_id)
    all_time_ids = sorted(train["time_id"].unique())
    cutoff = all_time_ids[int(len(all_time_ids) * 0.8) - 1]

    # Feature scaling (fit on early train data only)
    train_early = train[train["time_id"] <= cutoff]
    scaler = StandardScaler()
    scaler.fit(train_early[feature_cols].values)

    train_scaled = train.copy()
    train_scaled[feature_cols] = scaler.transform(train[feature_cols].values)

    test_scaled = test.copy()
    test_scaled[feature_cols] = scaler.transform(test[feature_cols].values)

    # Build sequences
    X_train, y_train, X_val, y_val, idx_train, idx_val = make_sequences(
        train_scaled, SEQ_LEN, feature_cols, cutoff
    )

    # DataLoaders
    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Model
    n_features = X_train.shape[2]
    model = LSTMVolatilityModel(input_size=n_features).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # Training
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_lstm.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    # Load best model
    model.load_state_dict(torch.load("best_model_lstm.pt", map_location=DEVICE))

    # Predict on test set
    X_test = test_scaled[feature_cols].values.astype(np.float32)

    # Build test sequences (using entire test set context)
    X_test_seq, y_test_seq, _, _, test_indices, _ = make_sequences(
        test_scaled, SEQ_LEN, feature_cols, test_scaled["time_id"].max() + 1
    )

    # Get predictions
    test_dataset = SequenceDataset(X_test_seq, y_test_seq)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    _, pred_vol, actual_vol = evaluate(model, test_loader, criterion)

    inference_time = time.perf_counter() - t0

    # Build output DataFrame matching backend format
    results = pd.DataFrame(
        {
            "model": spec.name,
            "model_type": spec.model_type,
            "fold": fold,
            "stock_id": test.loc[test_indices, "stock_id"].values,
            "time_id": test.loc[test_indices, "time_id"].values,
            "pred_var": np.square(np.maximum(pred_vol, 0.0)),
            "pred_vol": np.maximum(pred_vol, 0.0),
            "actual_var": np.square(actual_vol),
            "actual_vol": actual_vol,
            "inference_ms": (inference_time * 1000) / len(pred_vol),
            "feature_cols": ",".join(feature_cols),
            "prediction_kind": "horizon_line",
            "forecast_seconds": "",
            "forecast_vol_path": "",
        }
    )

    return results
