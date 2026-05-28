from __future__ import annotations

import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from typing import Any
from uuid import uuid4

import pandas as pd

from .artifacts import utc_timestamp, write_run_artifacts
from .config import PIPELINE_VERSION, DataConfig, FeatureConfig, ModelSpec, RunConfig, normalize_stock_id
from .data import load_processed_stock
from .evaluation import make_time_id_folds, summarize_fold_metrics
from .feature_cache import load_cached_features, write_feature_cache
from .features import load_stock_features, usable_feature_columns
from .models import is_arch_family_model, model_availability_issue, run_garch_on_processed, run_model_for_fold
from .universe import build_universe


_LAST_FEATURE_BACKEND = "serial"
_LAST_GARCH_BACKEND = "serial"


def _resolved_feature_workers(data_config: DataConfig, n_stocks: int) -> int:
    requested = data_config.feature_workers
    if requested is None:
        requested = max(1, (os.cpu_count() or 2) - 1)
    return max(1, min(int(requested), max(1, n_stocks)))


def _compute_stock_features(payload: dict[str, Any]) -> tuple[str, pd.DataFrame]:
    stock = payload["stock"]
    data_config = DataConfig(**payload["data_config"])
    feature_config = FeatureConfig(**payload["feature_config"])
    return stock, load_stock_features(stock, data_config, feature_config)


def _process_executor_kwargs() -> dict[str, Any]:
    if "fork" not in multiprocessing.get_all_start_methods():
        return {}
    return {"mp_context": multiprocessing.get_context("fork")}


def _parallel_stock_features(
    payloads: list[dict[str, Any]],
    workers: int,
    executor_cls,
    executor_kwargs: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    frames_by_stock: dict[str, pd.DataFrame] = {}
    with executor_cls(max_workers=workers, **(executor_kwargs or {})) as executor:
        futures = {executor.submit(_compute_stock_features, payload): payload["stock"] for payload in payloads}
        for future in as_completed(futures):
            stock = futures[future]
            try:
                result_stock, frame = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed to compute features for {stock}: {exc}") from exc
            frames_by_stock[result_stock] = frame
    return frames_by_stock


def load_live_features(data_config: DataConfig, feature_config: FeatureConfig) -> pd.DataFrame:
    global _LAST_FEATURE_BACKEND
    stocks = tuple(normalize_stock_id(stock) for stock in data_config.stocks)
    if not stocks:
        _LAST_FEATURE_BACKEND = "none"
        return pd.DataFrame()

    workers = _resolved_feature_workers(data_config, len(stocks))
    if workers == 1 or len(stocks) == 1:
        _LAST_FEATURE_BACKEND = "serial"
        frames = [load_stock_features(stock, data_config, feature_config) for stock in stocks]
    else:
        payloads = [
            {
                "stock": stock,
                "data_config": asdict(replace(data_config, stocks=(stock,))),
                "feature_config": asdict(feature_config),
            }
            for stock in stocks
        ]
        try:
            frames_by_stock = _parallel_stock_features(
                payloads,
                workers,
                ProcessPoolExecutor,
                _process_executor_kwargs(),
            )
            _LAST_FEATURE_BACKEND = "processes"
        except Exception:
            frames_by_stock = _parallel_stock_features(payloads, workers, ThreadPoolExecutor)
            _LAST_FEATURE_BACKEND = "threads_fallback"
        frames = [frames_by_stock[stock] for stock in stocks]

    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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

    features = load_live_features(config.data, config.features)
    written_cache = write_feature_cache(config.data, config.features, features)
    if written_cache is not None:
        cache_status = f"{cache_status}; wrote cache: {written_cache}"
    processed_by_stock = {
        stock: load_processed_stock(stock, config.data)
        for stock in config.data.stocks
    } if needs_processed else {}
    return features, processed_by_stock, "live", cache_status


def _run_garch_for_spec(
    processed_by_stock: dict[str, pd.DataFrame],
    spec: ModelSpec,
    horizon: int,
    fold: int,
    test_ids: set[int],
) -> pd.DataFrame:
    global _LAST_GARCH_BACKEND
    processed_frames = list(processed_by_stock.values())
    if not processed_frames:
        return pd.DataFrame()

    n_jobs = int(spec.parameters.get("n_jobs", 1) or 1)
    if n_jobs == 1 or len(processed_frames) == 1:
        _LAST_GARCH_BACKEND = "serial"
        parts = [
            run_garch_on_processed(processed, spec, horizon, fold, test_ids)
            for processed in processed_frames
        ]
    else:
        from joblib import Parallel, delayed

        try:
            parts = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(run_garch_on_processed)(processed, spec, horizon, fold, test_ids)
                for processed in processed_frames
            )
            _LAST_GARCH_BACKEND = "processes"
        except Exception:
            parts = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(run_garch_on_processed)(processed, spec, horizon, fold, test_ids)
                for processed in processed_frames
            )
            _LAST_GARCH_BACKEND = "threads_fallback"
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
        "feature_workers": _resolved_feature_workers(config.data, len(config.data.stocks)),
        "feature_parallel_backend": _LAST_FEATURE_BACKEND,
        "garch_parallel_backend": _LAST_GARCH_BACKEND,
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
            data=DataConfig(stocks=stocks, max_time_ids_per_stock=max_time_ids, feature_workers=1),
            features=FeatureConfig(forecast_horizon=60, n_pca_components=8),
            models=(
                ModelSpec(name="HAR-RV", model_type="HAR-RV"),
                ModelSpec(name="Linear Regression", model_type="Linear Regression"),
                ModelSpec(name="Random Forest", model_type="Random Forest", parameters={"n_estimators": 20}),
            ),
        )
    )
