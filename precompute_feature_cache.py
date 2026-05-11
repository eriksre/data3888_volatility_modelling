from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from back_end.config import (
    ACF_WINDOWS,
    BOOK_WINDOWS,
    EWMA_LAMBDAS,
    INDIVIDUAL_PARQUET_DIR,
    PIPELINE_VERSION,
    RETURN_WINDOWS,
    DataConfig,
    FeatureConfig,
    normalize_stock_id,
)
from back_end.data import list_available_stocks, stock_path
from back_end.features import load_stock_features, usable_feature_columns


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "feature_cache"
DEFAULT_FORECAST_HORIZON = 30

NOTEBOOK_RETURN_WINDOWS = (
    5,
    10,
    20,
    30,
    45,
    60,
    90,
    120,
    180,
    240,
    300,
    330,
    350,
    380,
    400,
    410,
    420,
    430,
    440,
    450,
    480,
    500,
    540,
    560,
    569,
)
NOTEBOOK_ACF_WINDOWS = (30, 60, 120, 300, 420, 569)
NOTEBOOK_BOOK_WINDOWS = (30, 60, 120, 300, 420)
NOTEBOOK_EWMA_LAMBDAS = (0.70, 0.80, 0.90, 0.94, 0.97, 0.985)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute feature rows for every time_id of every stock. "
            "Outputs parquet shards plus a combined parquet cache for the app."
        )
    )
    parser.add_argument("--source-dir", type=Path, default=INDIVIDUAL_PARQUET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--forecast-horizon", type=int, default=DEFAULT_FORECAST_HORIZON)
    parser.add_argument("--max-time-ids-per-stock", type=int, default=None)
    parser.add_argument("--stocks", nargs="*", default=None, help="Optional stock ids, e.g. stock_0 1 27.")
    parser.add_argument("--stock-limit", type=int, default=None, help="Limit stocks for a quick test run.")
    parser.add_argument(
        "--notebook-grid",
        action="store_true",
        help="Use the wider feature grid and 30-second horizon from assignment_2_eriks.ipynb.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute shards that already exist.")
    parser.add_argument("--no-combined", action="store_true", help="Skip writing features_all_stocks.parquet.")
    return parser.parse_args()


def make_feature_config(args: argparse.Namespace) -> FeatureConfig:
    if args.notebook_grid:
        return FeatureConfig(
            forecast_horizon=30,
            return_windows=NOTEBOOK_RETURN_WINDOWS,
            acf_windows=NOTEBOOK_ACF_WINDOWS,
            book_windows=NOTEBOOK_BOOK_WINDOWS,
            ewma_lambdas=NOTEBOOK_EWMA_LAMBDAS,
        )
    return replace(FeatureConfig(), forecast_horizon=args.forecast_horizon)


def resolve_stocks(source_dir: Path, requested: list[str] | None, stock_limit: int | None) -> list[str]:
    if requested:
        stocks = sorted({normalize_stock_id(stock) for stock in requested})
    else:
        stocks = list_available_stocks(source_dir)
    if stock_limit is not None:
        stocks = stocks[: max(0, stock_limit)]
    if not stocks:
        raise ValueError(f"No stock parquet files found in {source_dir}")
    return stocks


def feature_cache_columns(feature_config: FeatureConfig) -> list[str]:
    empty = pd.DataFrame(columns=["stock_id", "time_id", "target_var", "target_vol"])
    return ["stock_id", "time_id", "target_var", "target_vol", *usable_feature_columns(empty, feature_config)]


def compute_one_stock(payload: dict[str, Any]) -> dict[str, Any]:
    stock = payload["stock"]
    source_dir = payload["source_dir"]
    shard_path = Path(payload["shard_path"])
    data_config = DataConfig(
        stocks=(stock,),
        source_dir=source_dir,
        max_time_ids_per_stock=payload["max_time_ids_per_stock"],
    )
    feature_config = FeatureConfig(**payload["feature_config"])

    started = perf_counter()
    features = load_stock_features(stock, data_config, feature_config)
    elapsed = perf_counter() - started

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(shard_path, index=False)

    return {
        "stock_id": stock,
        "rows": int(len(features)),
        "time_ids": int(features["time_id"].nunique()) if not features.empty else 0,
        "columns": int(features.shape[1]),
        "usable_features": int(len(usable_feature_columns(features, feature_config))),
        "elapsed_seconds": elapsed,
        "shard_path": str(shard_path),
        "source_path": str(stock_path(stock, source_dir)),
    }


def write_manifest(
    output_dir: Path,
    *,
    source_dir: Path,
    combined_path: Path | None,
    stocks: list[str],
    feature_config: FeatureConfig,
    results: list[dict[str, Any]],
    total_elapsed: float,
    workers: int,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "source_dir": str(source_dir),
        "combined_path": str(combined_path) if combined_path is not None else None,
        "n_stocks_requested": len(stocks),
        "n_stocks_written": len(results),
        "total_rows": int(sum(result["rows"] for result in results)),
        "total_time_ids": int(sum(result["time_ids"] for result in results)),
        "workers": workers,
        "total_elapsed_seconds": total_elapsed,
        "feature_config": asdict(feature_config),
        "stocks": results,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def combine_shards(shard_paths: list[Path], combined_path: Path) -> tuple[int, int]:
    frames = [pd.read_parquet(path) for path in shard_paths]
    if not frames:
        raise ValueError("No feature shards were available to combine.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["stock_id", "time_id"]).reset_index(drop=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(combined_path, index=False)
    return int(len(combined)), int(combined.shape[1])


def assert_existing_cache_matches(output_dir: Path, source_dir: Path, feature_config: FeatureConfig) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    old_source = Path(manifest.get("source_dir", "")).resolve()
    old_config = manifest.get("feature_config")
    requested_config = json.loads(json.dumps(asdict(feature_config)))
    if old_source == source_dir and old_config == requested_config:
        return
    raise ValueError(
        "Existing cache manifest does not match the requested source/config. "
        "Use --force to rebuild, or choose a different --output-dir."
    )


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    shard_dir = output_dir / "by_stock"
    combined_path = output_dir / "features_all_stocks.parquet"
    feature_config = make_feature_config(args)
    workers = max(1, args.workers)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    stocks = resolve_stocks(source_dir, args.stocks, args.stock_limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.force and output_dir.exists():
        manifest_path = output_dir / "manifest.json"
        shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists():
            manifest_path.unlink()
        if combined_path.exists():
            combined_path.unlink()
    elif any(shard_dir.glob("stock_*.parquet")):
        assert_existing_cache_matches(output_dir, source_dir, feature_config)

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Stocks: {len(stocks):,}")
    print(f"Workers: {workers}")
    print(f"Forecast horizon: {feature_config.forecast_horizon}")
    print("Feature grid: assignment_2_eriks.ipynb" if args.notebook_grid else "Feature grid: backend default")

    started = perf_counter()
    results: list[dict[str, Any]] = []
    pending_payloads: list[dict[str, Any]] = []

    for stock in stocks:
        shard_path = shard_dir / f"{stock}.parquet"
        if shard_path.exists() and not args.force:
            features = pd.read_parquet(shard_path, columns=["stock_id", "time_id"])
            result = {
                "stock_id": stock,
                "rows": int(len(features)),
                "time_ids": int(features["time_id"].nunique()) if not features.empty else 0,
                "columns": None,
                "usable_features": None,
                "elapsed_seconds": 0.0,
                "shard_path": str(shard_path),
                "source_path": str(stock_path(stock, source_dir)),
                "cached": True,
            }
            results.append(result)
            print(f"cached {stock}: rows={result['rows']:,} time_ids={result['time_ids']:,}")
            continue
        pending_payloads.append(
            {
                "stock": stock,
                "source_dir": str(source_dir),
                "shard_path": str(shard_path),
                "max_time_ids_per_stock": args.max_time_ids_per_stock,
                "feature_config": asdict(feature_config),
            }
        )

    if pending_payloads:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(compute_one_stock, payload): payload["stock"] for payload in pending_payloads}
            for future in as_completed(futures):
                stock = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f"failed {stock}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    raise
                results.append(result)
                print(
                    f"done {stock}: rows={result['rows']:,} "
                    f"time_ids={result['time_ids']:,} "
                    f"elapsed={result['elapsed_seconds']:.2f}s"
                )

    results.sort(key=lambda item: item["stock_id"])
    shard_paths = [Path(result["shard_path"]) for result in results]
    total_rows = sum(result["rows"] for result in results)

    if args.no_combined:
        written_combined_path = None
    else:
        print("Combining stock shards...")
        combined_rows, combined_cols = combine_shards(shard_paths, combined_path)
        written_combined_path = combined_path
        print(f"Wrote combined cache: {combined_path} rows={combined_rows:,} columns={combined_cols:,}")

    total_elapsed = perf_counter() - started
    write_manifest(
        output_dir,
        source_dir=source_dir,
        combined_path=written_combined_path,
        stocks=stocks,
        feature_config=feature_config,
        results=results,
        total_elapsed=total_elapsed,
        workers=workers,
    )

    print(f"Wrote manifest: {output_dir / 'manifest.json'}")
    print(f"Total rows: {total_rows:,}")
    print(f"Total elapsed: {total_elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
