from __future__ import annotations

import json
import time
from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, RidgeCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import MODEL_REGISTRY, ModelSpec, normalize_stock_id
from .features import resolve_manual_feature_labels


ARCH_FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "forecast": {}},
    "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "o": 0, "q": 1, "forecast": {"method": "simulation", "simulations": 200}},
    "GJR-GARCH(1,1)": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "forecast": {}},
}


def is_arch_family_model(model_type: str) -> bool:
    return model_type in ARCH_FAMILY_SPECS


def make_estimator(model_type: str, parameters: dict[str, Any] | None = None, pca_components: int | None = None):
    params = parameters or {}
    def with_pca(estimator):
        if pca_components is None:
            return estimator
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=int(pca_components), random_state=42)),
                ("reg", estimator),
            ]
        )

    if model_type == "Linear Regression":
        estimator = LinearRegression(**params)
        return with_pca(estimator) if pca_components is not None else Pipeline([("scaler", StandardScaler()), ("reg", estimator)])
    if model_type == "Ridge Regression":
        estimator = RidgeCV(alphas=np.logspace(-4, 4, 25), **params)
        return with_pca(estimator) if pca_components is not None else Pipeline([("scaler", StandardScaler()), ("reg", estimator)])
    if model_type == "LASSO":
        estimator = Lasso(alpha=params.pop("alpha", 0.01), max_iter=params.pop("max_iter", 5000), **params)
        return with_pca(estimator) if pca_components is not None else Pipeline([("scaler", StandardScaler()), ("reg", estimator)])
    if model_type == "Decision Tree":
        return with_pca(DecisionTreeRegressor(max_depth=params.pop("max_depth", 6), min_samples_leaf=params.pop("min_samples_leaf", 20), random_state=params.pop("random_state", 42), **params))
    if model_type == "Random Forest":
        return with_pca(RandomForestRegressor(n_estimators=params.pop("n_estimators", 60), n_jobs=params.pop("n_jobs", -1), random_state=params.pop("random_state", 42), **params))
    if model_type == "XGBoost":
        import xgboost as xgb

        return with_pca(xgb.XGBRegressor(
            n_estimators=params.pop("n_estimators", 180),
            learning_rate=params.pop("learning_rate", 0.05),
            max_depth=params.pop("max_depth", 4),
            subsample=params.pop("subsample", 0.8),
            colsample_bytree=params.pop("colsample_bytree", 0.8),
            random_state=params.pop("random_state", 42),
            n_jobs=params.pop("n_jobs", -1),
            **params,
        ))
    raise ValueError(f"Unsupported estimator model type: {model_type}")


def model_availability_issue(model_type: str) -> str | None:
    if model_type not in MODEL_REGISTRY:
        return f"{model_type} is not a supported backend model type."
    required_module = MODEL_REGISTRY[model_type].required_module
    if required_module and find_spec(required_module) is None:
        return f"{model_type} requested but the '{required_module}' package is unavailable."
    return None


def default_features_for_model(model_type: str, available_columns: list[str]) -> list[str] | None:
    if model_type == "HAR-RV":
        cols = ["RV_30", "RV_60", "RV_120", "RV_240", "RV_300", "RV_420", "RV_500"]
        return [col for col in cols if col in available_columns]
    return None


def choose_feature_columns(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    inherited_feature_mode: str,
    inherited_manual_features: tuple[str, ...],
) -> list[str]:
    available = list(X_train.columns)
    defaults = default_features_for_model(spec.model_type, available)
    if defaults:
        return defaults

    mode = spec.feature_mode or inherited_feature_mode
    if mode == "Manual":
        manual = spec.manual_features or inherited_manual_features
        selected = resolve_manual_feature_labels(tuple(manual), available)
        return selected or available
    if mode == "PCA":
        return available

    raise ValueError(f"Unsupported feature selection mode: {mode}")


def run_ml_model(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    fold: int,
    pca_components: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_type = "Linear Regression" if spec.model_type == "HAR-RV" else spec.model_type
    if pca_components is not None:
        pca_components = max(1, min(int(pca_components), len(feature_cols), len(train)))
    estimator = make_estimator(model_type, dict(spec.parameters), pca_components=pca_components)
    y_train_vol = np.sqrt(np.maximum(train["target_var"].to_numpy(dtype=float), 0.0))
    estimator.fit(train[feature_cols], y_train_vol)

    start = time.perf_counter()
    pred_vol = np.maximum(estimator.predict(test[feature_cols]), 0.0)
    inference_ms = (time.perf_counter() - start) * 1000 / max(len(test), 1)
    pred_var = pred_vol**2
    output_feature_cols = [f"PC{i}" for i in range(1, pca_components + 1)] if pca_components is not None else feature_cols
    predictions = pd.DataFrame(
        {
            "model": spec.name,
            "model_type": spec.model_type,
            "fold": fold,
            "stock_id": test["stock_id"].values,
            "time_id": test["time_id"].values,
            "pred_var": pred_var,
            "pred_vol": pred_vol,
            "actual_var": test["target_var"].to_numpy(dtype=float),
            "actual_vol": test["target_vol"].to_numpy(dtype=float),
            "inference_ms": inference_ms,
            "feature_cols": ",".join(output_feature_cols),
            "prediction_kind": "horizon_line",
            "forecast_seconds": "",
            "forecast_vol_path": "",
        }
    )
    importance = extract_feature_importance(estimator, output_feature_cols, spec.name, fold)
    return predictions, importance


def extract_feature_importance(estimator, feature_cols: list[str], model_name: str, fold: int) -> pd.DataFrame:
    raw = None
    final_estimator = estimator
    if isinstance(estimator, Pipeline):
        final_estimator = estimator.steps[-1][1]
    if hasattr(final_estimator, "feature_importances_"):
        raw = final_estimator.feature_importances_
    elif hasattr(final_estimator, "coef_"):
        raw = np.ravel(final_estimator.coef_)
    if raw is None or len(raw) != len(feature_cols):
        return pd.DataFrame(columns=["model", "fold", "feature", "importance"])
    return pd.DataFrame(
        {
            "model": model_name,
            "fold": fold,
            "feature": feature_cols,
            "importance": np.abs(raw),
        }
    ).sort_values("importance", ascending=False)


def run_garch_on_processed(processed: pd.DataFrame, spec: ModelSpec, horizon: int, fold: int, test_ids: set[int]) -> pd.DataFrame:
    from arch import arch_model

    parameters = dict(spec.parameters)
    arch_spec = ARCH_FAMILY_SPECS[spec.model_type]
    model_kwargs = {
        key: value
        for key, value in arch_spec.items()
        if key != "forecast"
    }
    model_kwargs.update(parameters.get("model_kwargs", {}))
    forecast_kwargs = dict(arch_spec.get("forecast", {}))
    forecast_kwargs.update(parameters.get("forecast_kwargs", {}))
    fit_kwargs = {
        "disp": "off",
        "show_warning": False,
        **parameters.get("fit_kwargs", {}),
    }
    rows = []
    for time_id, chunk in processed[processed["time_id"].isin(test_ids)].groupby("time_id", sort=False):
        chunk = chunk.sort_values("seconds_in_bucket")
        raw_returns = chunk["log_price_diff"].to_numpy(dtype=float) * 10000
        seconds = chunk["seconds_in_bucket"].to_numpy(dtype=int)
        valid = np.isfinite(raw_returns)
        returns = raw_returns[valid]
        seconds = seconds[valid]
        if len(returns) < horizon + 10:
            continue
        cutoff = len(returns) - horizon
        train_ret = returns[:cutoff]
        test_ret = returns[cutoff:]
        forecast_seconds = seconds[cutoff:].tolist()
        inference_ms = np.nan
        try:
            start = time.perf_counter()
            model = arch_model(train_ret, mean="Zero", rescale=False, **model_kwargs)
            fit = model.fit(**fit_kwargs)
            forecast = fit.forecast(horizon=horizon, reindex=False, **forecast_kwargs)
            inference_ms = (time.perf_counter() - start) * 1000
            pred_path_var_arr = np.asarray(forecast.variance.values[-1], dtype=float)
            pred = float(np.mean(pred_path_var_arr))
        except Exception:
            pred = np.nan
            pred_path_var_arr = np.asarray([], dtype=float)
        actual = float(np.mean(test_ret**2))
        raw_stock_id = chunk["stock_id"].dropna().iloc[0]
        if isinstance(raw_stock_id, (float, np.floating)) and float(raw_stock_id).is_integer():
            raw_stock_id = int(raw_stock_id)
        stock_name = normalize_stock_id(raw_stock_id)
        pred_path_vol = np.sqrt(np.maximum(pred_path_var_arr, 0.0)).tolist()
        rows.append(
            {
                "model": spec.name,
                "model_type": spec.model_type,
                "fold": fold,
                "stock_id": stock_name,
                "time_id": int(time_id),
                "pred_var": pred,
                "pred_vol": np.sqrt(max(pred, 0.0)) if np.isfinite(pred) else np.nan,
                "actual_var": actual,
                "actual_vol": np.sqrt(max(actual, 0.0)),
                "inference_ms": inference_ms,
                "feature_cols": "",
                "prediction_kind": "garch_path",
                "forecast_seconds": json.dumps(forecast_seconds),
                "forecast_vol_path": json.dumps(pred_path_vol),
            }
        )
    return pd.DataFrame(rows)


def run_model_for_fold(
    spec: ModelSpec,
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold: int,
    inherited_feature_mode: str,
    inherited_n_pca_components: int,
    inherited_manual_features: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    issue = model_availability_issue(spec.model_type)
    if issue is not None:
        raise ValueError(issue)
    mode = spec.feature_mode or inherited_feature_mode
    n_components = spec.n_pca_components or inherited_n_pca_components
    feature_cols = choose_feature_columns(
        spec,
        train.drop(columns=["stock_id", "time_id", "target_var", "target_vol"]),
        inherited_feature_mode,
        inherited_manual_features,
    )
    return run_ml_model(
        spec,
        train,
        test,
        feature_cols,
        fold,
        pca_components=n_components if mode == "PCA" else None,
    )
