from __future__ import annotations

from dataclasses import replace
from typing import Any
from uuid import uuid4

import pandas as pd

from .artifacts import utc_timestamp, write_run_artifacts
from .config import PIPELINE_VERSION, DataConfig, FeatureConfig, ModelSpec, RunConfig, normalize_stock_id
from .data import load_processed_stock
from .evaluation import make_time_id_folds, summarize_fold_metrics
from .feature_cache import load_cached_features
from .features import load_stock_features, usable_feature_columns
from .models import is_arch_family_model, model_availability_issue, run_garch_on_processed, run_model_for_fold
from .universe import build_universe


def _normalize_config(config: RunConfig) -> RunConfig:
    stocks = tuple(normalize_stock_id(stock) for stock in config.data.stocks)
    data_config = replace(config.data, stocks=stocks)
    return replace(config, data=data_config)


def _load_feature_inputs(config: RunConfig) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], str, str | None]:
    needs_processed = any(is_arch_family_model(spec.model_type) for spec in config.models)
    cached, cache_status = load_cached_features(config.data, config.features)
    if cached is not None:
        processed_by_stock = {
            stock: load_processed_stock(stock, config.data)
            for stock in config.data.stocks
        } if needs_processed else {}
        return cached, processed_by_stock, "cache", cache_status

    frames = []
    processed_by_stock = {}
    for stock in config.data.stocks:
        frame = load_stock_features(stock, config.data, config.features)
        frames.append(frame)
        if needs_processed:
            processed_by_stock[stock] = load_processed_stock(stock, config.data)
    if not frames:
        return pd.DataFrame(), processed_by_stock, "live", cache_status
    return pd.concat(frames, ignore_index=True), processed_by_stock, "live", cache_status


def _run_garch_for_spec(
    processed_by_stock: dict[str, pd.DataFrame],
    spec: ModelSpec,
    horizon: int,
    fold: int,
    test_ids: set[int],
) -> pd.DataFrame:
    processed_frames = list(processed_by_stock.values())
    if not processed_frames:
        return pd.DataFrame()

    n_jobs = int(spec.parameters.get("n_jobs", 1) or 1)
    if n_jobs == 1 or len(processed_frames) == 1:
        parts = [
            run_garch_on_processed(processed, spec, horizon, fold, test_ids)
            for processed in processed_frames
        ]
    else:
        from joblib import Parallel, delayed

        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_garch_on_processed)(processed, spec, horizon, fold, test_ids)
            for processed in processed_frames
        )
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _run_models(
    config: RunConfig,
    features: pd.DataFrame,
    processed_by_stock: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), []

    feature_cols = usable_feature_columns(features, config.features)
    if not feature_cols:
        raise ValueError(
            "No usable feature columns were produced. Reduce forecast horizon, reduce lookback windows, "
            "or include more time windows."
        )

    data = features[["stock_id", "time_id", "target_var", "target_vol", *feature_cols]].copy()
    folds = make_time_id_folds(data["time_id"], config.split)
    prediction_parts = []
    importance_parts = []

    for split in folds:
        fold = int(split["fold"])
        train_ids = split["train_ids"]
        test_ids = split["test_ids"]
        train = data[data["time_id"].isin(train_ids)].copy()
        test = data[data["time_id"].isin(test_ids)].copy()
        if train.empty or test.empty:
            continue

        for spec in config.models:
            issue = model_availability_issue(spec.model_type)
            if issue is not None:
                raise ValueError(issue)
            if is_arch_family_model(spec.model_type):
                garch_predictions = _run_garch_for_spec(
                    processed_by_stock,
                    spec,
                    config.features.forecast_horizon,
                    fold,
                    test_ids,
                )
                if not garch_predictions.empty:
                    prediction_parts.append(garch_predictions)
                continue

            predictions, importance = run_model_for_fold(
                spec,
                train,
                test,
                fold,
                config.features.feature_mode,
                config.features.n_pca_components,
                config.features.manual_features,
            )
            if not predictions.empty:
                prediction_parts.append(predictions)
            if not importance.empty:
                importance_parts.append(importance)

    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    if predictions.empty:
        raise ValueError("No model predictions were produced. Check requested model types and data scope.")
    metrics = summarize_fold_metrics(predictions)
    importance = (
        pd.concat(importance_parts, ignore_index=True)
        if importance_parts
        else pd.DataFrame(columns=["model", "fold", "feature", "importance"])
    )
    return predictions, metrics, importance, []


def run_pipeline(config: RunConfig | None = None) -> dict[str, Any]:
    config = _normalize_config(config or RunConfig())
    run_id = f"{utc_timestamp()}_{uuid4().hex[:8]}"

    features, processed_by_stock, feature_source, feature_cache_status = _load_feature_inputs(config)
    if features.empty:
        raise ValueError(
            "No feature rows were produced. This usually means the forecast horizon is too long for the "
            "configured lookback windows or the selected stock scope is too small."
        )

    requested_model_issues = sorted(
        {
            issue
            for issue in (model_availability_issue(spec.model_type) for spec in config.models)
            if issue is not None
        }
    )
    if requested_model_issues:
        raise ValueError("; ".join(requested_model_issues))
    predictions, metrics, feature_importance, _ = _run_models(config, features, processed_by_stock)

    if config.universe.enabled:
        universe_summary, similarity = build_universe(features, predictions)
    else:
        universe_summary = similarity = pd.DataFrame()

    status = {
        "run_id": run_id,
        "status": "completed",
        "pipeline_version": PIPELINE_VERSION,
        "n_stocks": len(config.data.stocks),
        "n_feature_rows": int(len(features)),
        "n_predictions": int(len(predictions)),
        "train_pct": int(config.split.train_pct),
        "n_folds": int(config.split.n_folds),
        "feature_source": feature_source,
        "feature_cache_status": feature_cache_status,
    }
    directory = write_run_artifacts(
        run_id,
        config,
        status,
        predictions,
        metrics,
        feature_importance,
        universe_summary,
        similarity,
    )
    status["artifact_dir"] = str(directory)
    return status


def run_smoke_pipeline(stocks: tuple[str, ...] = ("stock_0",), max_time_ids: int = 40) -> dict[str, Any]:
    return run_pipeline(
        RunConfig(
            data=DataConfig(stocks=stocks, max_time_ids_per_stock=max_time_ids),
            features=FeatureConfig(forecast_horizon=60, n_pca_components=8),
            models=(
                ModelSpec(name="HAR-RV", model_type="HAR-RV"),
                ModelSpec(name="Linear Regression", model_type="Linear Regression"),
                ModelSpec(name="Random Forest", model_type="Random Forest", parameters={"n_estimators": 20}),
            ),
        )
    )
