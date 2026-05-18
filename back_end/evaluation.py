from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EPS, SplitConfig


def make_time_id_folds(time_ids: pd.Series, split_config: SplitConfig) -> list[dict[str, object]]:
    unique_ids = np.asarray(sorted(pd.Series(time_ids).dropna().unique()))
    if len(unique_ids) < 2:
        raise ValueError("At least two time_ids are required for a train/test split.")

    n_folds = min(max(2, int(split_config.n_folds)), len(unique_ids))
    fold_ids = unique_ids.copy()
    if split_config.shuffle:
        rng = np.random.default_rng(int(split_config.random_state))
        rng.shuffle(fold_ids)

    folds = []
    for fold, test_ids in enumerate(np.array_split(fold_ids, n_folds), start=1):
        if len(test_ids) == 0:
            continue
        train_ids = np.setdiff1d(fold_ids, test_ids, assume_unique=True)
        if len(train_ids) == 0:
            continue
        folds.append(
            {
                "fold": fold,
                "train_ids": set(train_ids.tolist()),
                "test_ids": set(test_ids.tolist()),
            }
        )
    return folds


def compute_metrics(predictions: pd.DataFrame, model_name: str) -> dict[str, float | int | str]:
    empty = {
        "model": model_name,
        "n": 0,
        "mse": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "rmspe": np.nan,
        "qlike": np.nan,
        "pearson_r": np.nan,
    }
    if predictions.empty:
        return empty

    pred = predictions["pred_var"].to_numpy(dtype=float)
    actual = predictions["actual_var"].to_numpy(dtype=float)
    finite = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[finite]
    actual = actual[finite]
    if len(pred) == 0:
        return empty

    pred_safe = np.maximum(pred, EPS)
    actual_safe = np.maximum(actual, EPS)
    mse = float(np.mean((pred - actual) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(pred - actual)))
    mape = float(np.mean(np.abs((pred_safe - actual_safe) / actual_safe)) * 100)
    rmspe = float(np.sqrt(np.mean(((pred_safe - actual_safe) / actual_safe) ** 2)))
    ratio = actual_safe / pred_safe
    qlike = float(np.mean(ratio - np.log(ratio) - 1))
    pearson = float(np.corrcoef(pred, actual)[0, 1]) if len(pred) > 1 and np.std(pred) > 0 and np.std(actual) > 0 else np.nan
    return {
        "model": model_name,
        "n": int(len(pred)),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "rmspe": rmspe,
        "qlike": qlike,
        "pearson_r": pearson,
    }


def summarize_fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, fold), chunk in predictions.groupby(["model", "fold"], dropna=False):
        rows.append({"fold": int(fold), **compute_metrics(chunk, str(model))})
    overall = []
    for model, chunk in predictions.groupby("model", dropna=False):
        overall.append({"fold": 0, **compute_metrics(chunk, str(model))})
    return pd.DataFrame(overall + rows)


def qlike_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pred_safe = np.clip(y_pred, EPS, None)
    realized_safe = np.clip(y_true, EPS, None)
    ratio = realized_safe / pred_safe
    return float(np.mean(ratio - np.log(ratio) - 1))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


AVAILABLE_METRICS = {
    "QLIKE": qlike_score,
    "RMSE": rmse_score,
    "MAE": mae_score,
}


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics_to_run: list[str] | None = None,
) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    metrics = metrics_to_run if metrics_to_run is not None else list(AVAILABLE_METRICS.keys())

    if len(y_true_clean) == 0:
        return {metric: np.nan for metric in metrics}

    results = {}
    for metric_name in metrics:
        metric_func = AVAILABLE_METRICS.get(metric_name.upper())
        if metric_func is None:
            raise ValueError(f"Performance Metric '{metric_name}' is not supported.")
        results[metric_name.upper()] = metric_func(y_true_clean, y_pred_clean)
    return results


def evaluate_rv_baseline(
    returns_lookup: dict[int, np.ndarray],
    time_ids: list[int],
    lookback_window: int = 570,
    target_horizon: int = 30,
) -> dict[str, float]:
    rows = []
    for tid in time_ids:
        returns = returns_lookup.get(tid)
        if returns is None:
            continue

        cutoff = len(returns) - target_horizon
        if cutoff < lookback_window:
            continue

        pred_var = float(np.mean(returns[cutoff - lookback_window:cutoff] ** 2))
        target_var = float(np.mean(returns[cutoff:] ** 2))
        if np.isfinite(pred_var) and np.isfinite(target_var):
            rows.append({"target_var": target_var, "pred_var": pred_var})

    df_forecast = pd.DataFrame(rows)
    if df_forecast.empty:
        return {metric: np.nan for metric in AVAILABLE_METRICS.keys()}

    return compute_evaluation_metrics(
        y_true=df_forecast["target_var"].to_numpy(),
        y_pred=df_forecast["pred_var"].to_numpy(),
    )
