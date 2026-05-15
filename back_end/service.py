from __future__ import annotations

from dataclasses import fields
import json
from typing import Any

import pandas as pd

import numpy as np

from .artifacts import latest_run_id, load_run_config, load_run_frame
from .config import (
    DataConfig,
    FeatureConfig,
    FRONTEND_MODEL_METRICS,
    SUPPORTED_MODEL_TYPES,
    RunConfig,
    SplitConfig,
    UniverseConfig,
    model_catalog,
    model_specs_from_ui,
    normalize_stock_id,
)
from .data import load_processed_stock, list_available_stocks
from .evaluation import compute_metrics
from .feature_cache import load_cached_features
from .features import load_stock_features
from .models import model_availability_issue
from .pipeline import run_pipeline
from .universe import LOSS_METRICS, build_model_comparison, build_pca_variance_explained, build_stock_pca


def available_stocks(source_dir: str | None = None) -> list[str]:
    data_config = DataConfig() if source_dir is None else DataConfig(source_dir=source_dir)
    return list_available_stocks(data_config.source_dir)


def available_model_catalog() -> list[dict[str, Any]]:
    available_types = [
        model_type
        for model_type in SUPPORTED_MODEL_TYPES
        if model_availability_issue(model_type) is None
    ]
    return model_catalog(available_types)


def start_run_from_ui(
    model_entries: list[dict[str, Any]],
    *,
    stocks: list[str] | tuple[str, ...] | None = None,
    forecast_horizon: int | None = None,
    max_time_ids_per_stock: int | None = None,
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("Add at least one model before running the pipeline.")
    selected_stocks = tuple(normalize_stock_id(stock) for stock in (stocks or available_stocks()))
    if not selected_stocks:
        raise ValueError("No stock parquet files were found to run.")

    statuses = []
    grouped_entries: dict[int, list[dict[str, Any]]] = {}

    for entry in model_entries:
        horizon = int(forecast_horizon or entry.get("pred_seconds", 30))
        grouped_entries.setdefault(horizon, []).append(entry)

    for horizon, entries in grouped_entries.items():
        specs = model_specs_from_ui(entries)
        issues = [
            issue
            for spec in specs
            for issue in [model_availability_issue(spec.model_type)]
            if issue is not None
        ]
        if issues:
            raise ValueError("; ".join(sorted(set(issues))))

        config = RunConfig(
            data=DataConfig(
                stocks=selected_stocks,
                max_time_ids_per_stock=max_time_ids_per_stock,
            ),
            features=FeatureConfig(forecast_horizon=horizon),
            split=SplitConfig(),
            models=specs,
            universe=UniverseConfig(enabled=True),
        )
        status = run_pipeline(config)
        statuses.append(status)

    if not statuses:
        raise ValueError("No runnable backend model groups were produced from the UI model list.")

    if len(statuses) == 1:
        return statuses[0]

    return {
        "run_id": statuses[-1]["run_id"],
        "status": "completed",
        "grouped_run": True,
        "run_ids": [status["run_id"] for status in statuses],
        "groups": statuses,
        "n_feature_rows": sum(int(status.get("n_feature_rows", 0)) for status in statuses),
        "n_predictions": sum(int(status.get("n_predictions", 0)) for status in statuses),
    }


def get_latest_or_selected_run(run_id: str | None = None) -> str | None:
    return run_id or latest_run_id()


def load_metrics(run_id: str | None = None) -> pd.DataFrame:
    selected = get_latest_or_selected_run(run_id)
    return load_run_frame(selected, "metrics.parquet") if selected else pd.DataFrame()


def load_predictions(run_id: str | None = None) -> pd.DataFrame:
    selected = get_latest_or_selected_run(run_id)
    return load_run_frame(selected, "predictions.parquet") if selected else pd.DataFrame()


def load_universe_summary(run_id: str | None = None) -> pd.DataFrame:
    selected = get_latest_or_selected_run(run_id)
    return load_run_frame(selected, "universe_summary.parquet") if selected else pd.DataFrame()


def load_similarity(run_id: str | None = None) -> pd.DataFrame:
    selected = get_latest_or_selected_run(run_id)
    return load_run_frame(selected, "stock_similarity.parquet") if selected else pd.DataFrame()


def prediction_series(run_id: str | None, stock_id: str, time_id: int | None = None) -> pd.DataFrame:
    predictions = load_predictions(run_id)
    if predictions.empty:
        return predictions
    subset = predictions[predictions["stock_id"] == normalize_stock_id(stock_id)]
    if time_id is not None:
        subset = subset[subset["time_id"] == time_id]
    return subset.sort_values(["model", "fold", "time_id"]).reset_index(drop=True)


def _json_list(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        if not value:
            return []
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return []
        return loaded if isinstance(loaded, list) else []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return []


def load_window_realized_series(run_id: str | None, stock_id: str, time_id: int) -> pd.DataFrame:
    selected = get_latest_or_selected_run(run_id)
    if not selected:
        return pd.DataFrame()

    config = load_run_config(selected)
    horizon = int(config.get("features", {}).get("forecast_horizon", 60))
    data_config = config.get("data", {})
    source_dir = data_config.get("source_dir")
    max_time_ids = data_config.get("max_time_ids_per_stock")
    processed = load_processed_stock(
        normalize_stock_id(stock_id),
        DataConfig(source_dir=source_dir, max_time_ids_per_stock=max_time_ids) if source_dir else DataConfig(max_time_ids_per_stock=max_time_ids),
    )
    chunk = processed[processed["time_id"] == time_id].sort_values("seconds_in_bucket").copy()
    if chunk.empty:
        return pd.DataFrame()

    returns = chunk["log_price_diff"].to_numpy(dtype=float) * 10000
    seconds = chunk["seconds_in_bucket"].to_numpy(dtype=int)
    valid = np.isfinite(returns)
    returns = returns[valid]
    seconds = seconds[valid]
    if len(returns) <= horizon:
        return pd.DataFrame()

    cutoff_idx = len(returns) - horizon
    split_second = int(seconds[cutoff_idx])
    rolling_window = 30
    rolling_vol = (
        pd.Series(returns ** 2)
        .rolling(rolling_window, min_periods=5)
        .sum()
        .pow(0.5)
        .to_numpy()
    )
    observed_vol = rolling_vol[:cutoff_idx]
    forecast_vol = rolling_vol[cutoff_idx:]
    observed = pd.DataFrame({
        "seconds_in_bucket": seconds[:cutoff_idx],
        "realized_vol": observed_vol,
        "segment": "observed",
        "split_second": split_second,
        "forecast_horizon": horizon,
    })
    heldout = pd.DataFrame({
        "seconds_in_bucket": seconds[cutoff_idx:],
        "realized_vol": forecast_vol,
        "segment": "heldout_actual",
        "split_second": split_second,
        "forecast_horizon": horizon,
    })
    return pd.concat([observed, heldout], ignore_index=True).dropna(subset=["realized_vol"])


def build_prediction_curves(realized_series: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if realized_series.empty or predictions.empty:
        return pd.DataFrame(columns=["model", "model_type", "seconds_in_bucket", "pred_vol", "prediction_kind"])

    heldout = realized_series[realized_series["segment"] == "heldout_actual"]
    if heldout.empty:
        return pd.DataFrame(columns=["model", "model_type", "seconds_in_bucket", "pred_vol", "prediction_kind"])

    seconds = heldout["seconds_in_bucket"].astype(int).tolist()
    rows = []
    for _, row in predictions.iterrows():
        model = str(row.get("model", "Model"))
        model_type = str(row.get("model_type", model))
        kind = str(row.get("prediction_kind", "horizon_line"))
        if kind == "garch_path":
            path_seconds = [int(x) for x in _json_list(row.get("forecast_seconds"))]
            path_vol = [float(x) for x in _json_list(row.get("forecast_vol_path"))]
            if path_seconds and path_vol:
                for second, pred_vol in zip(path_seconds, path_vol):
                    if np.isfinite(pred_vol):
                        rows.append(
                            {
                                "model": model,
                                "model_type": model_type,
                                "seconds_in_bucket": second,
                                "pred_vol": pred_vol,
                                "prediction_kind": "garch_path",
                            }
                        )
                continue

        pred_vol = row.get("pred_vol")
        if pred_vol is None or not np.isfinite(float(pred_vol)):
            continue
        for second in seconds:
            rows.append(
                {
                    "model": model,
                    "model_type": model_type,
                    "seconds_in_bucket": second,
                    "pred_vol": float(pred_vol),
                    "prediction_kind": "horizon_line",
                }
            )
    return pd.DataFrame(rows)


def load_stock_metrics(run_id: str | None, stock_id: str) -> pd.DataFrame:
    predictions = prediction_series(run_id, stock_id)
    if predictions.empty:
        return pd.DataFrame()
    rows = [compute_metrics(chunk, str(model)) for model, chunk in predictions.groupby("model")]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def load_individual_model_metrics(run_id: str | None, stock_id: str, include_scaffold: bool = True) -> pd.DataFrame:
    """Return rows shaped like front_end.pages.individual.MODEL_METRICS."""
    metrics = load_stock_metrics(run_id, stock_id)
    predictions = prediction_series(run_id, stock_id)
    use_scaffold = include_scaffold and metrics.empty and predictions.empty
    scaffold = pd.DataFrame(FRONTEND_MODEL_METRICS) if use_scaffold else pd.DataFrame(
        columns=["model", "inference_us", "rmse", "qlike"]
    )

    if not metrics.empty:
        for _, row in metrics.iterrows():
            model = str(row["model"])
            idx = scaffold.index[scaffold["model"] == model]
            if len(idx) == 0:
                scaffold = pd.concat(
                    [
                        scaffold,
                        pd.DataFrame([{
                            "model": model,
                            "inference_us": np.nan,
                            "rmse": np.nan,
                            "qlike": np.nan,
                        }]),
                    ],
                    ignore_index=True,
                )
                idx = scaffold.index[scaffold["model"] == model]
            scaffold.loc[idx, "rmse"] = row.get("rmse", np.nan)
            scaffold.loc[idx, "qlike"] = row.get("qlike", np.nan)

    if not predictions.empty:
        inference_us = predictions.groupby("model")["inference_ms"].mean() * 1000
        for model, value in inference_us.items():
            idx = scaffold.index[scaffold["model"] == model]
            if len(idx) and pd.notna(value):
                scaffold.loc[idx, "inference_us"] = value

    order = {row["model"]: idx for idx, row in enumerate(FRONTEND_MODEL_METRICS)}
    scaffold["_order"] = scaffold["model"].map(order).fillna(len(order))
    for col in ["inference_us", "rmse", "qlike"]:
        scaffold[col] = pd.to_numeric(scaffold[col], errors="coerce")
    return scaffold.sort_values(["_order", "model"]).drop(columns=["_order"]).reset_index(drop=True)


def load_individual_page_data(run_id: str | None, stock_id: str, time_id: int | None = None) -> dict[str, Any]:
    """Return data shaped for the current Individual Stock scaffold."""
    predictions = prediction_series(run_id, stock_id)
    time_ids = sorted(predictions["time_id"].unique().tolist()) if not predictions.empty else list(range(1, 11))
    selected_time_id = int(time_id or (time_ids[0] if time_ids else 1))
    selected_predictions = predictions[predictions["time_id"] == selected_time_id].reset_index(drop=True) if not predictions.empty else pd.DataFrame()
    realized_series = load_window_realized_series(run_id, stock_id, selected_time_id)
    return {
        "model_metrics": load_individual_model_metrics(run_id, stock_id),
        "stock_predictions": predictions,
        "predictions": selected_predictions,
        "prediction_curves": build_prediction_curves(realized_series, selected_predictions),
        "realized_series": realized_series,
        "time_ids": time_ids,
        "selected_time_id": selected_time_id,
    }


def load_pca_variance_explained(
    n_components: int,
    stocks: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    selected_stocks = tuple(normalize_stock_id(stock) for stock in (stocks or available_stocks()))
    if not selected_stocks:
        return pd.DataFrame(columns=["component", "explained_variance_ratio"])

    data_config = DataConfig(stocks=selected_stocks)
    feature_config = FeatureConfig()
    cached, _ = load_cached_features(data_config, feature_config)
    if cached is not None:
        features = cached
    else:
        frames = [load_stock_features(stock, data_config, feature_config) for stock in selected_stocks]
        frames = [frame for frame in frames if not frame.empty]
        features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return build_pca_variance_explained(features, feature_config, n_components)


def _dataclass_from_payload(cls, payload: dict[str, Any]):
    defaults = cls()
    values = {}
    for field in fields(cls):
        value = payload.get(field.name, getattr(defaults, field.name))
        if isinstance(getattr(defaults, field.name), tuple):
            value = tuple(value)
        values[field.name] = value
    return cls(**values)


def _load_run_features(run_id: str | None) -> tuple[pd.DataFrame, FeatureConfig]:
    selected = get_latest_or_selected_run(run_id)
    if not selected:
        return pd.DataFrame(), FeatureConfig()

    config = load_run_config(selected)
    data_config = _dataclass_from_payload(DataConfig, config.get("data", {}))
    feature_config = _dataclass_from_payload(FeatureConfig, config.get("features", {}))

    cached, _ = load_cached_features(data_config, feature_config)
    if cached is not None:
        return cached, feature_config

    frames = [load_stock_features(stock, data_config, feature_config) for stock in data_config.stocks]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(), feature_config
    return pd.concat(frames, ignore_index=True), feature_config


def _backfill_universe_summary_loss_metrics(summary: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or predictions.empty:
        return summary

    missing_metrics = [metric for metric in LOSS_METRICS if metric in summary.columns and summary[metric].isna().any()]
    if not missing_metrics:
        return summary

    rows = []
    for (stock_id, model), chunk in predictions.groupby(["stock_id", "model"], dropna=False):
        rows.append({"stock_id": stock_id, **compute_metrics(chunk, str(model))})
    if not rows:
        return summary

    stock_model_metrics = pd.DataFrame(rows)
    best_metrics = (
        stock_model_metrics.sort_values(["stock_id", "rmse"])
        .groupby("stock_id", as_index=False)
        .first()
        .rename(columns={"model": "best_model"})
    )
    fill_cols = ["stock_id", "best_model", *missing_metrics]
    backfill = best_metrics[[col for col in fill_cols if col in best_metrics.columns]]
    filled = summary.merge(backfill, on=["stock_id", "best_model"], how="left", suffixes=("", "_backfill"))
    for metric in missing_metrics:
        backfill_col = f"{metric}_backfill"
        if backfill_col in filled.columns:
            filled[metric] = filled[metric].fillna(filled[backfill_col])
            filled = filled.drop(columns=backfill_col)
    return filled


def load_universe_page_data(run_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[float], pd.DataFrame]:
    """Return data frames expected by the Universe scaffold."""
    summary = load_universe_summary(run_id)
    similarity = load_similarity(run_id)
    expected = ["stock_id", "mean_volatility", *LOSS_METRICS, "best_model"]
    if summary.empty:
        return pd.DataFrame(columns=expected), pd.DataFrame(), pd.DataFrame(), [], pd.DataFrame()
    for col in expected:
        if col not in summary.columns:
            summary[col] = np.nan if col != "best_model" else ""
    predictions = load_predictions(run_id)
    summary = _backfill_universe_summary_loss_metrics(summary, predictions)
    features, feature_config = _load_run_features(run_id)
    pca_df, pca_explained = build_stock_pca(features, feature_config)
    model_comparison = build_model_comparison(predictions, load_metrics(run_id))
    return summary[expected].copy(), similarity.copy(), pca_df, pca_explained, model_comparison
