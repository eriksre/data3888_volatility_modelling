from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from .config import UniverseConfig
from .evaluation import compute_metrics


def build_similarity(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    pivot = features.pivot_table(index="time_id", columns="stock_id", values="target_var")
    corr = pivot.corr().fillna(0.0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def cluster_from_similarity(similarity: pd.DataFrame, universe_config: UniverseConfig) -> pd.DataFrame:
    if similarity.empty:
        return pd.DataFrame(columns=["stock_id", "cluster"])
    stocks = list(similarity.index)
    n_clusters = min(max(1, universe_config.n_clusters), len(stocks))
    if n_clusters == 1:
        labels = np.zeros(len(stocks), dtype=int)
    else:
        clipped = similarity.clip(-1, 1)
        distance_values = (1 - clipped).to_numpy(copy=True)
        np.fill_diagonal(distance_values, 0.0)
        distance = pd.DataFrame(distance_values, index=similarity.index, columns=similarity.columns)
        try:
            model = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average")
        labels = model.fit_predict(distance)
    return pd.DataFrame({"stock_id": stocks, "cluster": labels.astype(int)})


def build_universe_summary(features: pd.DataFrame, predictions: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["stock_id", "mean_volatility", "rmse", "qlike", "best_model", "cluster"])
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
    if not clusters.empty:
        summary = summary.merge(clusters, on="stock_id", how="left")
    else:
        summary["cluster"] = np.nan
    return summary


def build_universe(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    universe_config: UniverseConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    similarity = build_similarity(features)
    clusters = cluster_from_similarity(similarity, universe_config)
    summary = build_universe_summary(features, predictions, clusters)
    return summary, similarity, clusters
