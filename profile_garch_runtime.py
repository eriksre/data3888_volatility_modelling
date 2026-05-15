from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from back_end.config import DataConfig, FeatureConfig, ModelSpec, RunConfig, SplitConfig, UniverseConfig
from back_end.pipeline import run_pipeline


DEFAULT_STOCKS = (
    "stock_0",
    "stock_1",
    "stock_2",
    "stock_3",
    "stock_4",
    "stock_5",
    "stock_6",
    "stock_7",
    "stock_8",
    "stock_9",
)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_model_types(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _read_prediction_summary(artifact_dir: str) -> dict[str, float | int]:
    predictions = pd.read_parquet(Path(artifact_dir) / "predictions.parquet")
    return {
        "prediction_rows": int(len(predictions)),
        "mean_fit_ms": float(predictions["inference_ms"].mean()),
        "median_fit_ms": float(predictions["inference_ms"].median()),
        "p95_fit_ms": float(predictions["inference_ms"].quantile(0.95)),
    }


def run_case(
    *,
    model_type: str,
    stocks: tuple[str, ...],
    max_time_ids: int,
    horizon: int,
    n_folds: int,
    n_jobs: int,
) -> dict[str, object]:
    config = RunConfig(
        data=DataConfig(stocks=stocks, max_time_ids_per_stock=max_time_ids),
        features=FeatureConfig(forecast_horizon=horizon),
        split=SplitConfig(n_folds=n_folds, train_pct=80),
        models=(
            ModelSpec(
                name=f"{model_type} n_jobs={n_jobs}",
                model_type=model_type,
                parameters={"n_jobs": n_jobs},
            ),
        ),
        universe=UniverseConfig(enabled=False),
    )
    start = perf_counter()
    status = run_pipeline(config)
    elapsed = perf_counter() - start
    summary = _read_prediction_summary(status["artifact_dir"])
    return {
        "model_type": model_type,
        "n_stocks": len(stocks),
        "max_time_ids_per_stock": max_time_ids,
        "forecast_horizon": horizon,
        "n_folds": n_folds,
        "n_jobs": n_jobs,
        "elapsed_seconds": elapsed,
        "artifact_dir": status["artifact_dir"],
        **summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GARCH runtime through the backend pipeline.")
    parser.add_argument("--stocks", default=",".join(DEFAULT_STOCKS), help="Comma-separated stock ids.")
    parser.add_argument("--max-time-ids", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--folds", type=int, default=1)
    parser.add_argument("--model-types", default="GARCH(1,1)", help="Semicolon-separated GARCH-family model types.")
    parser.add_argument("--n-jobs", default="1,4,8", help="Comma-separated worker counts.")
    parser.add_argument("--output-json", help="Optional path to write benchmark results as JSON.")
    args = parser.parse_args()

    stocks = _parse_csv(args.stocks)
    model_types = _parse_model_types(args.model_types)
    n_jobs_values = tuple(int(value) for value in _parse_csv(args.n_jobs))

    results = []
    for model_type in model_types:
        for n_jobs in n_jobs_values:
            result = run_case(
                model_type=model_type,
                stocks=stocks,
                max_time_ids=args.max_time_ids,
                horizon=args.horizon,
                n_folds=args.folds,
                n_jobs=n_jobs,
            )
            results.append(result)
            print(
                f"{model_type} n_jobs={n_jobs}: "
                f"{result['elapsed_seconds']:.3f}s total, "
                f"{result['prediction_rows']:,} predictions, "
                f"{result['mean_fit_ms']:.2f} ms/fit mean, "
                f"artifact={result['artifact_dir']}"
            )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
