from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT_DIR / "artifacts" / "runs"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "plots" / "report_figures"


def info(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate static report figures from backend run artifacts under "
            "artifacts/runs/<run_id>."
        )
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run id to use. If omitted, the latest completed run is used.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing run subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write PNG figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG DPI.",
    )
    return parser.parse_args()


def load_status(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        warn(f"Could not parse status file {path}: {exc}")
        return {}


def resolve_run_dir(runs_dir: Path, requested_run_id: str | None) -> Path | None:
    if not runs_dir.exists():
        warn(f"Runs directory does not exist: {runs_dir}")
        return None

    if requested_run_id:
        candidate = runs_dir / requested_run_id
        if candidate.exists() and candidate.is_dir():
            info(f"Using requested run: {requested_run_id}")
            return candidate
        warn(f"Requested run id was not found: {requested_run_id}")
        return None

    run_dirs = sorted([path for path in runs_dir.iterdir() if path.is_dir()], reverse=True)
    if not run_dirs:
        warn(f"No run directories were found in {runs_dir}")
        return None

    for run_dir in run_dirs:
        status = load_status(run_dir / "status.json")
        if str(status.get("status", "")).lower() == "completed":
            info(f"Using latest completed run: {run_dir.name}")
            return run_dir

    for run_dir in run_dirs:
        if (run_dir / "metrics.parquet").exists() and (run_dir / "predictions.parquet").exists():
            warn(
                "No run with status='completed' was found. Falling back to latest run with "
                f"metrics/predictions artifacts: {run_dir.name}"
            )
            return run_dir

    warn("No usable run artifacts were found.")
    return None


def read_parquet_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"Missing artifact file: {path.name}")
        return None
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        warn(f"Failed to read {path.name}: {exc}")
        return None


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    info(f"Saved {output_path}")


def model_metrics_frame(metrics: pd.DataFrame | None, predictions: pd.DataFrame | None) -> pd.DataFrame:
    rmse_rows = pd.DataFrame(columns=["model", "rmse"])
    qlike_rows = pd.DataFrame(columns=["model", "qlike"])

    if metrics is not None and not metrics.empty and "model" in metrics.columns:
        frame = metrics.copy()
        if "fold" in frame.columns and (frame["fold"] == 0).any():
            frame = frame[frame["fold"] == 0].copy()
        else:
            aggregations: dict[str, str] = {}
            if "rmse" in frame.columns:
                aggregations["rmse"] = "mean"
            if "qlike" in frame.columns:
                aggregations["qlike"] = "mean"
            if aggregations:
                frame = frame.groupby("model", as_index=False).agg(aggregations)

        if "rmse" in frame.columns:
            rmse_rows = frame[["model", "rmse"]].copy()
        if "qlike" in frame.columns:
            qlike_rows = frame[["model", "qlike"]].copy()

    if rmse_rows.empty and predictions is not None and not predictions.empty:
        required = {"model", "pred_var", "actual_var"}
        if required.issubset(predictions.columns):
            tmp = predictions[["model", "pred_var", "actual_var"]].copy()
            tmp["pred_var"] = pd.to_numeric(tmp["pred_var"], errors="coerce")
            tmp["actual_var"] = pd.to_numeric(tmp["actual_var"], errors="coerce")
            tmp = tmp.dropna(subset=["pred_var", "actual_var"])
            if not tmp.empty:
                rmse_rows = (
                    tmp.groupby("model", as_index=False)
                    .apply(lambda g: pd.Series({"rmse": float(np.sqrt(np.mean((g["pred_var"] - g["actual_var"]) ** 2)))}))
                    .reset_index(drop=True)
                )

    out = rmse_rows.merge(qlike_rows, on="model", how="outer")
    return out.sort_values("model").reset_index(drop=True) if not out.empty else out


def figure_accuracy_latency_tradeoff(
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    output_dir: Path,
    dpi: int,
) -> bool:
    if predictions is None or predictions.empty:
        warn("Skipping accuracy_latency_tradeoff.png: predictions are unavailable.")
        return False
    required = {"model", "inference_ms"}
    if not required.issubset(predictions.columns):
        warn("Skipping accuracy_latency_tradeoff.png: predictions are missing model/inference_ms.")
        return False

    model_metrics = model_metrics_frame(metrics, predictions)
    if model_metrics.empty or "rmse" not in model_metrics.columns:
        warn("Skipping accuracy_latency_tradeoff.png: RMSE values are unavailable.")
        return False

    latency = (
        predictions[["model", "inference_ms"]]
        .assign(inference_ms=lambda d: pd.to_numeric(d["inference_ms"], errors="coerce"))
        .dropna(subset=["inference_ms"])
        .groupby("model", as_index=False)["inference_ms"]
        .mean()
        .rename(columns={"inference_ms": "mean_inference_ms"})
    )
    frame = model_metrics.merge(latency, on="model", how="inner")
    frame = frame.dropna(subset=["rmse", "mean_inference_ms"])
    if frame.empty:
        warn("Skipping accuracy_latency_tradeoff.png: no model has both RMSE and latency.")
        return False

    use_microseconds = float(frame["mean_inference_ms"].max()) < 1.0
    x_values = frame["mean_inference_ms"] * (1000.0 if use_microseconds else 1.0)
    x_label = "Mean Inference Time (microseconds)" if use_microseconds else "Mean Inference Time (milliseconds)"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x_values, frame["rmse"], s=90, alpha=0.85)
    for _, row in frame.assign(x_plot=x_values).iterrows():
        ax.annotate(str(row["model"]), (row["x_plot"], row["rmse"]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMSE")
    ax.set_title("Accuracy-Latency Trade-off")
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir / "accuracy_latency_tradeoff.png", dpi)
    return True


def figure_model_metric_comparison(
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    output_dir: Path,
    dpi: int,
) -> bool:
    frame = model_metrics_frame(metrics, predictions)
    if frame.empty or "rmse" not in frame.columns:
        warn("Skipping model_metric_comparison.png: RMSE values are unavailable.")
        return False

    frame["rmse"] = pd.to_numeric(frame["rmse"], errors="coerce")
    if "qlike" in frame.columns:
        frame["qlike"] = pd.to_numeric(frame["qlike"], errors="coerce")
    frame = frame.sort_values("rmse", na_position="last")

    has_qlike = "qlike" in frame.columns and frame["qlike"].notna().any()
    if has_qlike:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax_rmse, ax_qlike = axes
        ax_rmse.bar(frame["model"], frame["rmse"], color="#4e79a7")
        ax_rmse.set_title("RMSE by Model")
        ax_rmse.set_ylabel("RMSE")
        ax_rmse.tick_params(axis="x", rotation=30)
        ax_rmse.grid(axis="y", alpha=0.2)

        qlike_frame = frame.dropna(subset=["qlike"]).copy()
        ax_qlike.bar(qlike_frame["model"], qlike_frame["qlike"], color="#f28e2b")
        ax_qlike.set_title("QLIKE by Model")
        ax_qlike.set_ylabel("QLIKE")
        ax_qlike.tick_params(axis="x", rotation=30)
        ax_qlike.grid(axis="y", alpha=0.2)
        fig.suptitle("Model Metric Comparison")
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(frame["model"], frame["rmse"], color="#4e79a7")
        ax.set_title("Model RMSE Comparison")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.2)

    save_figure(fig, output_dir / "model_metric_comparison.png", dpi)
    return True


def choose_representative_stock_time(predictions: pd.DataFrame) -> tuple[str, int] | None:
    required = {"stock_id", "time_id", "model"}
    if not required.issubset(predictions.columns):
        return None
    group = (
        predictions.groupby(["stock_id", "time_id"], as_index=False)
        .agg(n_models=("model", "nunique"), n_rows=("model", "size"))
        .sort_values(["n_models", "n_rows", "stock_id", "time_id"], ascending=[False, False, True, True])
    )
    if group.empty:
        return None
    row = group.iloc[0]
    return str(row["stock_id"]), int(row["time_id"])


def figure_single_stock_actual_vs_predicted(
    predictions: pd.DataFrame | None,
    output_dir: Path,
    dpi: int,
) -> bool:
    if predictions is None or predictions.empty:
        warn("Skipping single_stock_actual_vs_predicted.png: predictions are unavailable.")
        return False

    required = {"stock_id", "time_id", "model", "actual_vol", "pred_vol"}
    if not required.issubset(predictions.columns):
        warn("Skipping single_stock_actual_vs_predicted.png: predictions are missing required columns.")
        return False

    selected = choose_representative_stock_time(predictions)
    if selected is None:
        warn("Skipping single_stock_actual_vs_predicted.png: could not pick a representative stock/time window.")
        return False
    stock_id, time_id = selected

    subset = predictions[
        (predictions["stock_id"].astype(str) == stock_id)
        & (pd.to_numeric(predictions["time_id"], errors="coerce") == float(time_id))
    ].copy()
    subset["actual_vol"] = pd.to_numeric(subset["actual_vol"], errors="coerce")
    subset["pred_vol"] = pd.to_numeric(subset["pred_vol"], errors="coerce")
    subset = subset.dropna(subset=["actual_vol", "pred_vol"])
    if subset.empty:
        warn("Skipping single_stock_actual_vs_predicted.png: selected stock/time has no finite values.")
        return False

    by_model = (
        subset.groupby("model", as_index=False)
        .agg(actual_vol=("actual_vol", "mean"), pred_vol=("pred_vol", "mean"))
        .sort_values("model")
    )

    x = np.arange(len(by_model))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(by_model) + 4), 5.5))
    ax.bar(x - width / 2, by_model["actual_vol"], width=width, label="Actual volatility", color="#4e79a7")
    ax.bar(x + width / 2, by_model["pred_vol"], width=width, label="Predicted volatility", color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(by_model["model"], rotation=30, ha="right")
    ax.set_ylabel("Volatility")
    ax.set_title(f"Single Stock Actual vs Predicted Volatility ({stock_id}, time_id={time_id})")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_dir / "single_stock_actual_vs_predicted.png", dpi)
    return True


def figure_universe_best_model_counts(
    universe_summary: pd.DataFrame | None,
    output_dir: Path,
    dpi: int,
) -> bool:
    if universe_summary is None or universe_summary.empty:
        warn("Skipping universe_best_model_counts.png: universe summary is unavailable.")
        return False
    if "best_model" not in universe_summary.columns:
        warn("Skipping universe_best_model_counts.png: best_model column is missing.")
        return False

    counts = (
        universe_summary["best_model"]
        .fillna("")
        .astype(str)
        .str.strip()
    )
    counts = counts[counts != ""].value_counts().sort_values(ascending=False)
    if counts.empty:
        warn("Skipping universe_best_model_counts.png: no non-empty best_model values found.")
        return False

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values, color="#59a14f")
    ax.set_title("Best Model Counts Across Stocks")
    ax.set_ylabel("Count of stocks")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom")
    save_figure(fig, output_dir / "universe_best_model_counts.png", dpi)
    return True


def figure_feature_importance_top10(
    feature_importance: pd.DataFrame | None,
    output_dir: Path,
    dpi: int,
) -> bool:
    if feature_importance is None or feature_importance.empty:
        warn("Skipping feature_importance_top10.png: feature importance is unavailable or empty.")
        return False
    required = {"feature", "importance"}
    if not required.issubset(feature_importance.columns):
        warn("Skipping feature_importance_top10.png: feature/importance columns are missing.")
        return False

    frame = feature_importance[["feature", "importance"]].copy()
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce")
    frame = frame.dropna(subset=["importance"])
    if frame.empty:
        warn("Skipping feature_importance_top10.png: no finite importance values found.")
        return False

    top10 = (
        frame.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(10)
        .sort_values("importance", ascending=True)
    )
    if top10.empty:
        warn("Skipping feature_importance_top10.png: top-10 set is empty.")
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top10["feature"], top10["importance"], color="#b07aa1")
    ax.set_title("Top 10 Average Feature Importances")
    ax.set_xlabel("Average importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, output_dir / "feature_importance_top10.png", dpi)
    return True


def main() -> int:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()

    run_dir = resolve_run_dir(runs_dir, args.run_id)
    if run_dir is None:
        return 1

    info(f"Reading artifacts from: {run_dir}")
    metrics = read_parquet_safe(run_dir / "metrics.parquet")
    predictions = read_parquet_safe(run_dir / "predictions.parquet")
    feature_importance = read_parquet_safe(run_dir / "feature_importance.parquet")
    universe_summary = read_parquet_safe(run_dir / "universe_summary.parquet")
    _ = read_parquet_safe(run_dir / "stock_similarity.parquet")

    figure_tasks: list[tuple[str, Callable[[], bool]]] = [
        (
            "accuracy_latency_tradeoff.png",
            lambda: figure_accuracy_latency_tradeoff(metrics, predictions, output_dir, args.dpi),
        ),
        (
            "model_metric_comparison.png",
            lambda: figure_model_metric_comparison(metrics, predictions, output_dir, args.dpi),
        ),
        (
            "single_stock_actual_vs_predicted.png",
            lambda: figure_single_stock_actual_vs_predicted(predictions, output_dir, args.dpi),
        ),
        (
            "universe_best_model_counts.png",
            lambda: figure_universe_best_model_counts(universe_summary, output_dir, args.dpi),
        ),
        (
            "feature_importance_top10.png",
            lambda: figure_feature_importance_top10(feature_importance, output_dir, args.dpi),
        ),
    ]

    generated = 0
    for name, task in figure_tasks:
        try:
            if task():
                generated += 1
        except Exception as exc:
            warn(f"Failed to generate {name}: {exc}")

    info(f"Generated {generated}/{len(figure_tasks)} report figures in {output_dir}")
    return 0 if generated > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

