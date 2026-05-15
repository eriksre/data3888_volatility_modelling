"""
Hyperparameter tuning for RF, XGBoost and LightGBM.

Loads pre-extracted features from utils/, runs RandomizedSearchCV with
temporal cross-validation, and writes the best parameters to
back_end/tuned_params.json.

Usage:
    python -m back_end.tuning

The output JSON can be passed directly into Erik's ModelSpec.parameters:
    ModelSpec(name="XGBoost", model_type="XGBoost",
              parameters=tuned_params["XGBoost"])
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# Make sure repo root is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import load_single_stock, temporal_split
from config import FEATURE_COLS, CLUSTER_CSV

OUTPUT_PATH = Path(__file__).parent / "tuned_params.json"
N_ITER      = 40   # random search iterations per model — increase for better results
CV_FOLDS    = 3    # inner CV folds during search
RANDOM_SEED = 42
N_STOCKS    = 20   # stocks to sample for tuning (more = slower but more robust)


# ── Parameter grids ────────────────────────────────────────────────────────────

RF_GRID = {
    "n_estimators":    [100, 200, 300, 500],
    "max_depth":       [None, 6, 10, 15, 20],
    "min_samples_leaf":[1, 2, 5, 10, 20],
    "max_features":    ["sqrt", "log2", 0.5, 0.7],
    "min_samples_split":[2, 5, 10],
}

XGB_GRID = {
    "n_estimators":    [100, 200, 300, 500],
    "learning_rate":   [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth":       [3, 4, 5, 6, 8],
    "subsample":       [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree":[0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight":[1, 3, 5, 10],
    "reg_alpha":       [0, 0.01, 0.1, 1.0],
    "reg_lambda":      [0.1, 1.0, 5.0, 10.0],
}

LGBM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
    "max_depth":        [3, 4, 5, 6, 8, -1],
    "num_leaves":       [15, 31, 50, 63, 100],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_samples":[5, 10, 20, 50],
    "reg_alpha":        [0, 0.01, 0.1, 1.0],
    "reg_lambda":       [0, 0.01, 0.1, 1.0],
}

DT_GRID = {
    "max_depth":        [3, 4, 5, 6, 8, 10, 15, None],
    "min_samples_leaf": [1, 2, 5, 10, 20, 50],
    "min_samples_split":[2, 5, 10, 20],
    "max_features":     ["sqrt", "log2", 0.5, 0.7, None],
}

LASSO_GRID = {
    "alpha": [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_tuning_data(n_stocks: int = N_STOCKS) -> tuple[np.ndarray, np.ndarray]:
    """Load features and targets from a sample of stocks for tuning."""
    cluster_df = pd.read_csv(CLUSTER_CSV)
    # sample evenly from both clusters
    c1 = cluster_df[cluster_df["cluster"] == 1]["stock_id"].tolist()
    c2 = cluster_df[cluster_df["cluster"] == 2]["stock_id"].tolist()
    rng = np.random.default_rng(RANDOM_SEED)
    n1  = max(1, n_stocks // 3)
    n2  = n_stocks - n1
    sample = list(rng.choice(c1, min(n1, len(c1)), replace=False)) + \
             list(rng.choice(c2, min(n2, len(c2)), replace=False))

    frames = []
    for sid in sample:
        try:
            df = load_single_stock(sid)
            if len(df) >= 100:
                frames.append(df)
        except Exception as e:
            print(f"  skipping stock {sid}: {e}")

    if not frames:
        raise RuntimeError("No stocks loaded — check DATA_DIR in config.py")

    data  = pd.concat(frames, ignore_index=True)
    train, _, _ = temporal_split(data)

    # drop rows with NaN features
    available = [c for c in FEATURE_COLS if c in train.columns]
    train = train.dropna(subset=available + ["log_rv_second"])

    X = train[available].values.astype(np.float32)
    y = train["log_rv_second"].values.astype(np.float32)
    print(f"  Tuning data: {len(X):,} rows, {X.shape[1]} features "
          f"({len(frames)} stocks)")
    return X, y


# ── Temporal CV splitter ───────────────────────────────────────────────────────

class TemporalKFold:
    """Simple walk-forward splitter that respects time order."""
    def __init__(self, n_splits: int = CV_FOLDS):
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(1, self.n_splits + 1):
            train_end = fold_size * i
            test_end  = min(fold_size * (i + 1), n)
            yield np.arange(0, train_end), np.arange(train_end, test_end)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ── RMSE scorer ───────────────────────────────────────────────────────────────

rmse_scorer = make_scorer(
    lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred)),
    greater_is_better=True,
)


# ── Tuning functions ──────────────────────────────────────────────────────────

def tune_model(name: str, estimator, param_grid: dict, X: np.ndarray, y: np.ndarray) -> dict:
    print(f"\n{'='*50}")
    print(f"Tuning {name}  ({N_ITER} iterations × {CV_FOLDS} folds)")
    print(f"{'='*50}")
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring=rmse_scorer,
        cv=TemporalKFold(CV_FOLDS),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
        refit=False,
    )
    search.fit(X, y)

    best = search.best_params_
    best_rmse = -search.best_score_
    elapsed = time.time() - t0

    print(f"\n  Best RMSE : {best_rmse:.4f}")
    print(f"  Time      : {elapsed/60:.1f} min")
    print(f"  Params    : {json.dumps(best, indent=4)}")
    return best


def run_tuning():
    print("Loading tuning data...")
    X, y = load_tuning_data()

    # Load existing results so already-tuned models are not re-run
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
        print(f"Loaded existing results for: {list(results.keys())}")
    else:
        results = {}

    # Random Forest
    if "Random Forest" not in results:
        results["Random Forest"] = tune_model(
            "Random Forest",
            RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            RF_GRID, X, y,
        )
    else:
        print("\nRandom Forest — already tuned, skipping.")

    # XGBoost
    if "XGBoost" not in results:
        try:
            from xgboost import XGBRegressor
            results["XGBoost"] = tune_model(
                "XGBoost",
                XGBRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbosity=0),
                XGB_GRID, X, y,
            )
        except ImportError:
            print("\nXGBoost not installed — skipping.")
    else:
        print("\nXGBoost — already tuned, skipping.")

    # LightGBM
    if "LightGBM" not in results:
        try:
            from lightgbm import LGBMRegressor
            results["LightGBM"] = tune_model(
                "LightGBM",
                LGBMRegressor(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1),
                LGBM_GRID, X, y,
            )
        except ImportError:
            print("\nLightGBM not installed — skipping.")
    else:
        print("\nLightGBM — already tuned, skipping.")

    # Decision Tree
    if "Decision Tree" not in results:
        from sklearn.tree import DecisionTreeRegressor
        results["Decision Tree"] = tune_model(
            "Decision Tree",
            DecisionTreeRegressor(random_state=RANDOM_SEED),
            DT_GRID, X, y,
        )
    else:
        print("\nDecision Tree — already tuned, skipping.")

    # LASSO
    if "LASSO" not in results:
        from sklearn.linear_model import Lasso
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        results["LASSO"] = tune_model(
            "LASSO",
            Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Lasso(max_iter=5000)),
            ]),
            {"reg__alpha": LASSO_GRID["alpha"]},
            X, y,
        )
    else:
        print("\nLASSO — already tuned, skipping.")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved tuned parameters to: {OUTPUT_PATH}")

    # Print usage instructions
    print("\n" + "="*50)
    print("To use in Erik's pipeline, pass params into ModelSpec:")
    print("="*50)
    for model_name, params in results.items():
        print(f"\n  ModelSpec(name=\"{model_name}\", model_type=\"{model_name}\",")
        print(f"            parameters={json.dumps(params)})")


if __name__ == "__main__":
    run_tuning()
