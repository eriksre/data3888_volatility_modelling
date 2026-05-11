from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import DataConfig, FeatureConfig, normalize_stock_id
from .features import expanded_feature_cols

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - parquet engine fallback
    pq = None


def feature_cache_path(data_config: DataConfig) -> Path | None:
    if not data_config.use_feature_cache or not data_config.feature_cache_path:
        return None
    path = Path(data_config.feature_cache_path)
    return path if path.exists() else None


def feature_cache_manifest(cache_path: Path) -> dict[str, object]:
    manifest_path = cache_path.parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}


def _manifest_horizon_matches(manifest: dict[str, object], feature_config: FeatureConfig) -> bool:
    manifest_config = manifest.get("feature_config")
    if not isinstance(manifest_config, dict):
        return True
    return int(manifest_config.get("forecast_horizon", -1)) == int(feature_config.forecast_horizon)


def _source_matches(manifest: dict[str, object], data_config: DataConfig) -> bool:
    source_dir = manifest.get("source_dir")
    if not source_dir:
        return True
    return Path(str(source_dir)).resolve() == Path(data_config.source_dir).resolve()


def parquet_columns(cache_path: Path) -> set[str]:
    if pq is not None:
        return set(pq.read_schema(cache_path).names)
    return set(pd.read_parquet(cache_path).columns)


def cache_compatible(cache_path: Path, data_config: DataConfig, feature_config: FeatureConfig) -> tuple[bool, str]:
    manifest = feature_cache_manifest(cache_path)
    if not manifest:
        return False, "cache manifest is missing or unreadable"
    if not _manifest_horizon_matches(manifest, feature_config):
        return False, "forecast horizon does not match cache manifest"
    if not _source_matches(manifest, data_config):
        return False, "source directory does not match cache manifest"

    try:
        cache_columns = parquet_columns(cache_path)
    except Exception as exc:
        return False, f"could not read cache columns: {exc}"

    required = {"stock_id", "time_id", "target_var", "target_vol"}
    missing_base = required.difference(cache_columns)
    if missing_base:
        return False, f"cache is missing required columns: {sorted(missing_base)}"

    missing_features = [col for col in expanded_feature_cols(feature_config) if col not in cache_columns]
    if missing_features:
        return False, f"cache is missing feature columns for this config: {missing_features[:5]}"

    return True, "ok"


def load_cached_features(data_config: DataConfig, feature_config: FeatureConfig) -> tuple[pd.DataFrame | None, str]:
    cache_path = feature_cache_path(data_config)
    if cache_path is None:
        return None, "feature cache disabled or missing"

    compatible, reason = cache_compatible(cache_path, data_config, feature_config)
    if not compatible:
        return None, reason

    stocks = [normalize_stock_id(stock) for stock in data_config.stocks]
    required_columns = ["stock_id", "time_id", "target_var", "target_vol", *expanded_feature_cols(feature_config)]
    cached = pd.read_parquet(cache_path, columns=required_columns)
    cached = cached[cached["stock_id"].isin(stocks)].copy()
    if cached.empty:
        return None, "cache contains none of the requested stocks"

    missing_stocks = sorted(set(stocks).difference(cached["stock_id"].unique()))
    if missing_stocks:
        return None, f"cache is missing requested stocks: {missing_stocks[:5]}"

    if data_config.max_time_ids_per_stock is not None:
        limit = int(data_config.max_time_ids_per_stock)
        cached = (
            cached.sort_values(["stock_id", "time_id"])
            .groupby("stock_id", group_keys=False)
            .head(limit)
            .reset_index(drop=True)
        )
    else:
        cached = cached.sort_values(["stock_id", "time_id"]).reset_index(drop=True)

    return cached, str(cache_path)
