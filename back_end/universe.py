from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import FeatureConfig
from .evaluation import compute_metrics
from .features import usable_feature_columns


def build_similarity(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    pivot = features.pivot_table(index="time_id", columns="stock_id", values="target_var")
    corr = pivot.corr().fillna(0.0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def build_universe_summary(features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["stock_id", "mean_volatility", "rmse", "qlike", "best_model"])
    summary = (
        features.groupby("stock_id", as_index=False)
        .agg(mean_volatility=("target_vol", "mean"), mean_variance=("target_var", "mean"), n_windows=("time_id", "nunique"))
    )
    if not predictions.empty:
        rows = []
        for (stock_id, model), chunk in predictions.groupby(["stock_id", "model"], dropna=False):
            rows.append({"stock_id": stock_id, **compute_metrics(chunk, str(model))})
        stock_model_metrics = pd.DataFrame(rows)
        best = (
            stock_model_metrics.sort_values(["stock_id", "rmse"])
            .groupby("stock_id", as_index=False)
            .first()[["stock_id", "model", "rmse", "qlike"]]
            .rename(columns={"model": "best_model"})
        )
        summary = summary.merge(best, on="stock_id", how="left")
    else:
        summary["best_model"] = ""
        summary["rmse"] = np.nan
        summary["qlike"] = np.nan
    return summary


def build_stock_pca(features: pd.DataFrame, feature_config: FeatureConfig) -> tuple[pd.DataFrame, list[float]]:
    if features.empty:
        return pd.DataFrame(columns=["stock_id"]), []

    feature_cols = usable_feature_columns(features, feature_config)
    if not feature_cols:
        return pd.DataFrame(columns=["stock_id"]), []

    stock_features = features.groupby("stock_id", as_index=True)[feature_cols].mean()
    stock_features = stock_features.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    stock_features = stock_features.loc[:, stock_features.nunique(dropna=True) > 1]
    if len(stock_features) < 2 or stock_features.shape[1] < 2:
        return pd.DataFrame(columns=["stock_id"]), []

    stock_features = stock_features.fillna(stock_features.mean())
    values = StandardScaler().fit_transform(stock_features)
    n_components = min(len(stock_features), stock_features.shape[1])
    model = PCA(n_components=n_components, random_state=42)
    transformed = model.fit_transform(values)
    columns = [f"PC{i}" for i in range(1, n_components + 1)]
    pca_df = pd.DataFrame(transformed, index=stock_features.index, columns=columns).reset_index()
    explained = model.explained_variance_ratio_.tolist()
    return pca_df, [float(value) for value in explained]


def build_pca_variance_explained(
    features: pd.DataFrame,
    feature_config: FeatureConfig,
    n_components: int,
) -> pd.DataFrame:
    columns = ["component", "explained_variance_ratio"]
    if features.empty:
        return pd.DataFrame(columns=columns)

    feature_cols = usable_feature_columns(features, feature_config)
    if not feature_cols:
        return pd.DataFrame(columns=columns)

    feature_matrix = features[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    feature_matrix = feature_matrix.loc[:, feature_matrix.nunique(dropna=True) > 1]
    if len(feature_matrix) < 2 or feature_matrix.shape[1] < 2:
        return pd.DataFrame(columns=columns)

    feature_matrix = feature_matrix.fillna(feature_matrix.mean())
    component_count = min(int(n_components), len(feature_matrix), feature_matrix.shape[1])
    if component_count < 1:
        return pd.DataFrame(columns=columns)

    values = StandardScaler().fit_transform(feature_matrix)
    model = PCA(n_components=component_count, random_state=42)
    model.fit(values)
    return pd.DataFrame(
        {
            "component": [f"PC{i}" for i in range(1, component_count + 1)],
            "explained_variance_ratio": [float(value) for value in model.explained_variance_ratio_],
        }
    )


def _feature_set_label(feature_cols: pd.Series) -> str:
    values = [str(value) for value in feature_cols.dropna().unique() if str(value)]
    if not values:
        return ""
    first = values[0]
    cols = [col for col in first.split(",") if col]
    if cols and all(col.startswith("PC") for col in cols):
        return f"{len(cols)} PCs"
    if cols:
        return f"{len(cols)} features"
    return ""


def build_model_comparison(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mse", "rmse", "mae", "mape", "rmspe", "qlike", "pearson_r"]
    winner_metrics = ["mse", "rmse", "mae", "mape", "rmspe", "qlike"]
    columns = [
        "model",
        "model_type",
        "feature_set",
        "n_predictions",
        "mean_inference_ms",
        "median_inference_ms",
        "p95_inference_ms",
        *metric_cols,
        *[f"fold_std_{metric}" for metric in winner_metrics],
        *[f"best_stocks_{metric}" for metric in winner_metrics],
        *[f"best_stock_share_{metric}" for metric in winner_metrics],
        *[f"avg_rank_{metric}" for metric in winner_metrics],
    ]
    if predictions.empty and metrics.empty:
        return pd.DataFrame(columns=columns)

    overall = metrics[metrics["fold"] == 0].copy() if "fold" in metrics.columns else metrics.copy()
    if overall.empty and not predictions.empty:
        overall = pd.DataFrame([compute_metrics(chunk, str(model)) for model, chunk in predictions.groupby("model")])

    if not predictions.empty:
        model_info = (
            predictions.groupby("model", as_index=False)
            .agg(
                model_type=("model_type", "first"),
                n_predictions=("pred_var", "count"),
                mean_inference_ms=("inference_ms", "mean"),
                median_inference_ms=("inference_ms", "median"),
                p95_inference_ms=("inference_ms", lambda values: values.quantile(0.95)),
                feature_set=("feature_cols", _feature_set_label),
            )
        )
    else:
        model_info = pd.DataFrame(columns=["model", "model_type", "n_predictions", "mean_inference_ms", "median_inference_ms", "p95_inference_ms", "feature_set"])

    comparison = overall[["model", *[col for col in metric_cols if col in overall.columns]]].merge(
        model_info,
        on="model",
        how="outer",
    )

    fold_metrics = metrics[metrics["fold"] != 0].copy() if "fold" in metrics.columns else pd.DataFrame()
    if not fold_metrics.empty:
        fold_std_metrics = [metric for metric in winner_metrics if metric in fold_metrics.columns]
        if fold_std_metrics:
            fold_std = fold_metrics.groupby("model")[fold_std_metrics].std().add_prefix("fold_std_").reset_index()
            comparison = comparison.merge(fold_std, on="model", how="left")

    if not predictions.empty:
        stock_rows = [
            {"stock_id": stock_id, **compute_metrics(chunk, str(model))}
            for (stock_id, model), chunk in predictions.groupby(["stock_id", "model"], dropna=False)
        ]
        stock_metrics = pd.DataFrame(stock_rows)
        n_stocks = int(stock_metrics["stock_id"].nunique()) if not stock_metrics.empty else 0
        for metric in winner_metrics:
            if metric not in stock_metrics.columns:
                continue
            ranked = stock_metrics.dropna(subset=[metric]).copy()
            if ranked.empty:
                continue
            ranked[f"rank_{metric}"] = ranked.groupby("stock_id")[metric].rank(method="min", ascending=True)
            best = ranked[ranked[f"rank_{metric}"] == 1].groupby("model").size().rename(f"best_stocks_{metric}")
            avg_rank = ranked.groupby("model")[f"rank_{metric}"].mean().rename(f"avg_rank_{metric}")
            comparison = comparison.merge(best, on="model", how="left")
            comparison = comparison.merge(avg_rank, on="model", how="left")
            comparison[f"best_stocks_{metric}"] = comparison[f"best_stocks_{metric}"].fillna(0).astype(int)
            comparison[f"best_stock_share_{metric}"] = comparison[f"best_stocks_{metric}"] / max(n_stocks, 1)

    for col in columns:
        if col not in comparison.columns:
            comparison[col] = np.nan
    return comparison[columns].sort_values(["rmse", "model"], na_position="last").reset_index(drop=True)


def build_universe(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    similarity = build_similarity(features)
    summary = build_universe_summary(features, predictions)
    return summary, similarity
