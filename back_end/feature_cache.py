from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import DataConfig, FeatureConfig, PIPELINE_VERSION, normalize_stock_id
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


def writable_feature_cache_path(data_config: DataConfig) -> Path | None:
    if not data_config.use_feature_cache or not data_config.feature_cache_path:
        return None
    return Path(data_config.feature_cache_path)


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


def _pipeline_version_matches(manifest: dict[str, object]) -> bool:
    return str(manifest.get("pipeline_version", "")) == PIPELINE_VERSION


def _source_matches(manifest: dict[str, object], data_config: DataConfig) -> bool:
    source_dir = manifest.get("source_dir")
    if not source_dir:
        return True
    return Path(str(source_dir)).resolve() == Path(data_config.source_dir).resolve()


def _time_id_limit_matches(manifest: dict[str, object], data_config: DataConfig) -> bool:
    cached_limit = manifest.get("max_time_ids_per_stock")
    if cached_limit is None:
        data = manifest.get("data_config")
        cached_limit = data.get("max_time_ids_per_stock") if isinstance(data, dict) else None
    if cached_limit is None:
        return True
    requested_limit = data_config.max_time_ids_per_stock
    return requested_limit is not None and int(requested_limit) <= int(cached_limit)


def parquet_columns(cache_path: Path) -> set[str]:
    if pq is not None:
        return set(pq.read_schema(cache_path).names)
    return set(pd.read_parquet(cache_path).columns)


def cache_compatible(cache_path: Path, data_config: DataConfig, feature_config: FeatureConfig) -> tuple[bool, str]:
    manifest = feature_cache_manifest(cache_path)
    if not manifest:
        return False, "cache manifest is missing or unreadable"
    if not _pipeline_version_matches(manifest):
        return False, "pipeline version does not match cache manifest"
    if not _manifest_horizon_matches(manifest, feature_config):
        return False, "forecast horizon does not match cache manifest"
    if not _source_matches(manifest, data_config):
        return False, "source directory does not match cache manifest"
    if not _time_id_limit_matches(manifest, data_config):
        return False, "time-id limit does not match cache manifest"

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


def write_feature_cache(data_config: DataConfig, feature_config: FeatureConfig, features: pd.DataFrame) -> str | None:
    cache_path = writable_feature_cache_path(data_config)
    if cache_path is None or features.empty:
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = features.sort_values(["stock_id", "time_id"]).reset_index(drop=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    ordered.to_parquet(tmp_path, index=False)
    tmp_path.replace(cache_path)

    stocks = sorted(str(stock) for stock in ordered["stock_id"].dropna().unique())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "source_dir": str(Path(data_config.source_dir).resolve()),
        "combined_path": str(cache_path.resolve()),
        "max_time_ids_per_stock": data_config.max_time_ids_per_stock,
        "n_stocks_requested": len(tuple(data_config.stocks)),
        "n_stocks_written": len(stocks),
        "total_rows": int(len(ordered)),
        "total_time_ids": int(ordered["time_id"].nunique()) if "time_id" in ordered else 0,
        "feature_config": asdict(feature_config),
        "data_config": asdict(data_config),
        "stocks": [
            {
                "stock_id": stock,
                "rows": int(len(chunk)),
                "time_ids": int(chunk["time_id"].nunique()) if "time_id" in chunk else 0,
            }
            for stock, chunk in ordered.groupby("stock_id", sort=True)
        ],
    }
    (cache_path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return str(cache_path)
