"""
LSTM model for short-term volatility forecasting.

Each stock's time_ids are treated as a time series. Sliding windows of
SEQ_LEN consecutive feature rows — including the current timestep — are
fed into a stacked LSTM which predicts log(RV) for that window's second half.

One model is trained on all 112 stocks combined (same scope as cluster_models.py)
using a strict temporal split — no future data ever touches training.

Key design note: the sequence window includes the current timestep so the
model has access to rv_first (first-half realized volatility) when predicting
log_rv_second — exactly the same information available to the tree-based models.

Usage:
    pip install torch
    python -m back_end.lstm

Output saved to plots/lstm/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import load_single_stock, temporal_split, qlike
from config import FEATURE_COLS, CLUSTER_CSV, PLOTS_DIR

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise ImportError("PyTorch not found. Install with:  pip install torch")

# ── Hyperparameters ────────────────────────────────────────────────────────────

SEQ_LEN       = 20     # consecutive windows fed as context to the LSTM
HIDDEN_SIZE   = 128    # units per LSTM layer
NUM_LAYERS    = 2      # stacked LSTM layers
DROPOUT       = 0.2    # dropout between layers and before output head
LEARNING_RATE = 5e-4
BATCH_SIZE    = 256
MAX_EPOCHS    = 100
PATIENCE      = 10     # early stopping — stop if val loss doesn't improve
N_STOCKS      = 112    # use all stocks
RANDOM_SEED   = 42

OUTPUT_DIR = Path(PLOTS_DIR) / "lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (n_samples, seq_len, n_features)
        # y: (n_samples,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ──────────────────────────────────────────────────────────────────────

class LSTMVolatilityModel(nn.Module):
    """
    Stacked LSTM with a two-layer MLP head and layer normalisation.

    The sequence window includes the current timestep, so the final hidden
    state encodes both recent history and the current window's first-half
    market microstructure features (spread, depth, imbalance, rv_first, etc.).
    """

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
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
        out    = self.layer_norm(out[:, -1, :])   # last timestep, normalised
        return self.head(out).squeeze(-1)


# ── Sequence construction ──────────────────────────────────────────────────────

def make_sequences(
    data: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str],
    cutoff_time_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding window sequences per stock.

    Each sequence spans [i-seq_len+1 ... i] (inclusive), so the final step
    in every window IS the current timestep — giving the model access to
    rv_first and all first-half microstructure features for the window being
    predicted.  This mirrors the feature matrix used by LightGBM/RF and is
    the correct setup for same-window volatility forecasting.

    Train/test split: sequences whose last time_id <= cutoff go to train,
    the rest go to test.
    """
    X_train, y_train = [], []
    X_test,  y_test  = [], []

    for _, group in data.groupby("stock_id"):
        group = group.sort_values("time_id").reset_index(drop=True)
        X_all = group[feature_cols].values.astype(np.float32)
        y_all = group["log_rv_second"].values.astype(np.float32)
        tids  = group["time_id"].values

        # i is the index of the CURRENT (target) timestep
        # window covers [i-seq_len+1 ... i]
        for i in range(seq_len - 1, len(X_all)):
            seq_X = X_all[i - seq_len + 1 : i + 1]  # (seq_len, n_features)
            seq_y = y_all[i]
            if tids[i] <= cutoff_time_id:
                X_train.append(seq_X)
                y_train.append(seq_y)
            else:
                X_test.append(seq_X)
                y_test.append(seq_y)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test),  np.array(y_test),
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
        loss  = criterion(preds, y_batch)
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
            preds       = model(X_batch)
            total_loss += criterion(preds, y_batch).item() * len(y_batch)
            all_preds.append(preds.cpu().numpy())
            all_actuals.append(y_batch.cpu().numpy())
    preds   = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    return total_loss / len(loader.dataset), preds, actuals


# ── Main training loop ─────────────────────────────────────────────────────────

def run_lstm():
    print("Loading data for all stocks...")
    t0 = time.time()

    cluster_df = pd.read_csv(CLUSTER_CSV)
    stock_ids  = sorted(cluster_df["stock_id"].tolist())

    frames = []
    for i, sid in enumerate(stock_ids[:N_STOCKS]):
        try:
            df = load_single_stock(sid)
            if len(df) >= SEQ_LEN + 5:
                frames.append(df)
            print(f"  [{i+1}/{N_STOCKS}] stock {sid} loaded ({len(df)} rows)")
        except Exception as e:
            print(f"  [{i+1}/{N_STOCKS}] stock {sid} skipped: {e}")

    data = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows: {len(data):,}  |  Stocks: {len(frames)}")

    # Temporal split — find global cutoff at 80th percentile of time_ids
    all_time_ids = sorted(data["time_id"].unique())
    cutoff = all_time_ids[int(len(all_time_ids) * 0.8) - 1]
    print(f"Temporal cutoff: time_id={cutoff}  "
          f"(train<={cutoff}, test>{cutoff})")

    # Feature scaling — fit on train rows only to prevent leakage
    available_features = [c for c in FEATURE_COLS if c in data.columns]
    train_rows = data[data["time_id"] <= cutoff]
    scaler     = StandardScaler()
    scaler.fit(train_rows[available_features].values)

    data_scaled = data.copy()
    data_scaled[available_features] = scaler.transform(
        data[available_features].values
    )

    # Build sequences
    print("\nBuilding sequences...")
    X_train, y_train, X_test, y_test = make_sequences(
        data_scaled, SEQ_LEN, available_features, cutoff
    )
    print(f"  Train sequences: {len(X_train):,}")
    print(f"  Test  sequences: {len(X_test):,}")

    # DataLoaders
    train_loader = DataLoader(
        SequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        SequenceDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    # Model
    n_features = X_train.shape[2]
    model      = LSTMVolatilityModel(input_size=n_features).to(DEVICE)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=1e-5)
    criterion  = nn.MSELoss()
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {DEVICE}")
    print(f"\nTraining  (max {MAX_EPOCHS} epochs, early stop patience={PATIENCE})")

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    history           = []

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss     = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, _, _ = evaluate(model, test_loader, criterion)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss})
        print(f"  Epoch {epoch:02d}/{MAX_EPOCHS}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # Load best weights and get final test metrics
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt",
                                     map_location=DEVICE))
    _, preds, actuals = evaluate(model, test_loader, criterion)

    # ── Metrics ────────────────────────────────────────────────────────────────
    errors     = actuals - preds
    abs_errors = np.abs(errors)
    eps        = 1e-8   # avoid division by zero

    rmse  = float(np.sqrt(mean_squared_error(actuals, preds)))
    rmspe = float(np.sqrt(np.mean((errors / (actuals + eps)) ** 2)))
    ql    = qlike(actuals, preds)
    mae   = float(mean_absolute_error(actuals, preds))
    mape  = float(np.mean(abs_errors / (np.abs(actuals) + eps)) * 100)

    # Huber loss (delta = 1.0): quadratic for small errors, linear for large
    delta      = 1.0
    huber_elem = np.where(abs_errors <= delta,
                          0.5 * errors ** 2,
                          delta * (abs_errors - 0.5 * delta))
    huber      = float(np.mean(huber_elem))

    elapsed = (time.time() - t0) / 60

    print(f"\n{'='*50}")
    print(f"LSTM Results")
    print(f"{'='*50}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  RMSPE : {rmspe:.4f}")
    print(f"  QLIKE : {ql:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  MAPE  : {mape:.4f}%")
    print(f"  Huber : {huber:.4f}")
    print(f"  Time  : {elapsed:.1f} min")

    # Save results
    results = {
        "rmse":           rmse,
        "rmspe":          rmspe,
        "qlike":          ql,
        "mae":            mae,
        "mape":           mape,
        "huber":          huber,
        "seq_len":        SEQ_LEN,
        "hidden_size":    HIDDEN_SIZE,
        "num_layers":     NUM_LAYERS,
        "dropout":        DROPOUT,
        "learning_rate":  LEARNING_RATE,
        "n_stocks":       len(frames),
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
        "epochs_trained": len(history),
    }
    with open(OUTPUT_DIR / "lstm_results.json", "w") as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    run_lstm()
