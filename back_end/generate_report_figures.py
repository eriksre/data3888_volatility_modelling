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
from matplotlib.ticker import MaxNLocator


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = ROOT_DIR / "artifacts" / "runs"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "plots" / "report_figures"


COLOR_THEME = {
    "ink": "#243b53",
    "actual_line": "#0b2545",
    "primary": "#2f4b7c",
    "secondary": "#577399",
    "pareto_line": "#1f5aa6",
    "accent": "#7a9eb1",
    "muted": "#7b8794",
    "grid": "#c7d3de",
    "pareto": "#b22222",
    "dominated": "#9aa5b1",
    "latency_1us": "#6f8f72",
    "latency_10us": "#b08d2d",
    "latency_1000us": "#7d4ea3",
    "feature": "#516f8d",
}

MODEL_COLOR_CYCLE = [
    "#E69F00",
    "#009E73",
    "#7E57C2",
    "#D55E00",
    "#4C78A8",
    "#CC79A7",
    "#56B4E9",
    "#8C8C8C",
]

MODEL_LINESTYLES = ["--", "-.", ":", (0, (4, 2)), (0, (3, 1, 1, 1)), (0, (2, 1))]

GLOBAL_MPL_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#7b8794",
    "axes.labelcolor": "#243b53",
    "axes.titlecolor": "#102a43",
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "font.size": 10,
    "font.family": ["DejaVu Sans"],
    "xtick.color": "#334e68",
    "ytick.color": "#334e68",
    "grid.color": "#c7d3de",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.facecolor": "white",
    "legend.edgecolor": "#d9e2ec",
    "legend.fontsize": 9,
    "savefig.facecolor": "white",
}


def apply_global_style() -> None:
    plt.style.use("default")
    matplotlib.rcParams.update(GLOBAL_MPL_STYLE)


def styled_subplots(
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (10.0, 6.0),
    constrained_layout: bool = True,
) -> tuple[plt.Figure, plt.Axes | np.ndarray]:
    return plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        constrained_layout=constrained_layout,
    )


def style_axis(ax: plt.Axes, *, axis: str = "both", which: str = "major", alpha: float = 0.35) -> None:
    ax.grid(True, axis=axis, which=which, alpha=alpha)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def model_color_map(models: list[str]) -> dict[str, str]:
    ordered = sorted({str(model) for model in models})
    return {model: MODEL_COLOR_CYCLE[idx % len(MODEL_COLOR_CYCLE)] for idx, model in enumerate(ordered)}


def annotate_lollipop_values(
    ax: plt.Axes,
    y_pos: np.ndarray,
    values: np.ndarray,
    *,
    fmt: str,
    pad_fraction: float = 0.02,
    fontsize: float = 9.0,
) -> None:
    if values.size == 0:
        return
    x_min = float(np.min(values))
    x_max = float(np.max(values))
    span = max(x_max - x_min, abs(x_max) * 0.1, 1e-9)
    pad = span * pad_fraction
    for y, x in zip(y_pos, values):
        ax.text(
            float(x + pad),
            float(y),
            format(float(x), fmt),
            va="center",
            ha="left",
            fontsize=fontsize,
            color=COLOR_THEME["ink"],
        )


def downsample_for_plot(frame: pd.DataFrame, *, max_points: int = 1500) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    idx = np.linspace(0, len(frame) - 1, num=max_points, dtype=int)
    return frame.iloc[np.unique(idx)]


def gentle_smooth(
    values: pd.Series,
    *,
    target_points: int,
    scale: int = 75,
    min_window: int = 3,
    max_window: int = 25,
) -> pd.Series:
    n = int(max(target_points, 1))
    #Keep smooth mild
    window = max(min_window, min(max_window, n // max(scale, 1)))
    if window % 2 == 0:
        window += 1
    return values.rolling(window=window, center=True, min_periods=1).mean()


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
        frame["model"] = frame["model"].astype(str)
        aggregations: dict[str, str] = {}
        if "rmse" in frame.columns:
            frame["rmse"] = pd.to_numeric(frame["rmse"], errors="coerce")
            aggregations["rmse"] = "mean"
        if "qlike" in frame.columns:
            frame["qlike"] = pd.to_numeric(frame["qlike"], errors="coerce")
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


def _aggregate_metric_from_predictions(
    predictions: pd.DataFrame,
    metric_name: str,
) -> pd.DataFrame:
    required = {"model", "pred_var", "actual_var"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame(columns=["model", metric_name])

    tmp = predictions[["model", "pred_var", "actual_var"]].copy()
    tmp["pred_var"] = pd.to_numeric(tmp["pred_var"], errors="coerce")
    tmp["actual_var"] = pd.to_numeric(tmp["actual_var"], errors="coerce")
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["pred_var", "actual_var"])
    if tmp.empty:
        return pd.DataFrame(columns=["model", metric_name])

    if metric_name == "rmse":
        out = (
            tmp.groupby("model", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {"rmse": float(np.sqrt(np.mean((g["pred_var"] - g["actual_var"]) ** 2)))}
                )
            )
            .reset_index(drop=True)
        )
        return out

    if metric_name == "qlike":
        eps = 1e-12

        def qlike_group(g: pd.DataFrame) -> float:
            pred = np.maximum(g["pred_var"].to_numpy(dtype=float), eps)
            actual = np.maximum(g["actual_var"].to_numpy(dtype=float), eps)
            ratio = actual / pred
            return float(np.mean(ratio - np.log(ratio) - 1.0))

        out = (
            tmp.groupby("model", as_index=False)
            .apply(lambda g: pd.Series({"qlike": qlike_group(g)}))
            .reset_index(drop=True)
        )
        return out

    return pd.DataFrame(columns=["model", metric_name])


def model_summary_frame(metrics: pd.DataFrame | None, predictions: pd.DataFrame | None) -> pd.DataFrame:
    summary = pd.DataFrame(columns=["model", "mean_inference_us", "qlike", "rmse"])

    if metrics is not None and not metrics.empty and "model" in metrics.columns:
        frame = metrics.copy()
        frame["model"] = frame["model"].astype(str)
        aggregations: dict[str, str] = {}
        if "qlike" in frame.columns:
            frame["qlike"] = pd.to_numeric(frame["qlike"], errors="coerce")
            aggregations["qlike"] = "mean"
        if "rmse" in frame.columns:
            frame["rmse"] = pd.to_numeric(frame["rmse"], errors="coerce")
            aggregations["rmse"] = "mean"
        if aggregations:
            frame = frame.groupby("model", as_index=False).agg(aggregations)

        keep_cols = ["model"]
        if "qlike" in frame.columns:
            keep_cols.append("qlike")
        if "rmse" in frame.columns:
            keep_cols.append("rmse")
        summary = frame[keep_cols].drop_duplicates(subset=["model"], keep="first").copy()

    if predictions is not None and not predictions.empty and "model" in predictions.columns:
        preds = predictions.copy()
        preds["model"] = preds["model"].astype(str)

        if "inference_ms" in preds.columns:
            latency = (
                preds[["model", "inference_ms"]]
                .assign(inference_ms=lambda d: pd.to_numeric(d["inference_ms"], errors="coerce"))
                .replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["inference_ms"])
                .groupby("model", as_index=False)["inference_ms"]
                .mean()
                .rename(columns={"inference_ms": "mean_inference_ms"})
            )
            if not latency.empty:
                latency["mean_inference_us"] = latency["mean_inference_ms"] * 1000.0
                latency = latency[["model", "mean_inference_us"]]
                summary = summary.merge(latency, on="model", how="outer")

        rmse_from_preds = _aggregate_metric_from_predictions(preds, "rmse")
        qlike_from_preds = _aggregate_metric_from_predictions(preds, "qlike")
        if not rmse_from_preds.empty:
            summary = summary.merge(
                rmse_from_preds.rename(columns={"rmse": "rmse_pred"}),
                on="model",
                how="outer",
            )
        if not qlike_from_preds.empty:
            summary = summary.merge(
                qlike_from_preds.rename(columns={"qlike": "qlike_pred"}),
                on="model",
                how="outer",
            )

    for col in ["mean_inference_us", "qlike", "rmse"]:
        if col not in summary.columns:
            summary[col] = np.nan
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    if "rmse_pred" in summary.columns:
        summary["rmse"] = summary["rmse"].fillna(summary["rmse_pred"])
        summary = summary.drop(columns=["rmse_pred"])
    if "qlike_pred" in summary.columns:
        summary["qlike"] = summary["qlike"].fillna(summary["qlike_pred"])
        summary = summary.drop(columns=["qlike_pred"])

    return summary.sort_values("model").reset_index(drop=True)


def pareto_optimal_mask(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
) -> pd.Series:
    values = frame[[x_col, y_col]].to_numpy(dtype=float)
    n = len(values)
    optimal = np.ones(n, dtype=bool)
    for i in range(n):
        xi, yi = values[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = values[j]
            if (xj <= xi and yj <= yi) and (xj < xi or yj < yi):
                optimal[i] = False
                break
    return pd.Series(optimal, index=frame.index)


def _label_offsets(n_points: int) -> list[tuple[int, int]]:
    base = [(10, 10), (10, -11), (14, 15), (14, -16), (18, 20), (18, -21), (22, 26), (22, -27)]
    return [base[i % len(base)] for i in range(max(n_points, 0))]


def _normalized_distance_to_ideal(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_norm = (x - x_min) / max(x_max - x_min, 1e-12)
    y_norm = (y - y_min) / max(y_max - y_min, 1e-12)
    dist = np.sqrt(x_norm**2 + y_norm**2)
    return pd.Series(dist, index=df.index)


def _safe_quantile(values: np.ndarray, q: float, method: str = "linear") -> float:
    try:
        return float(np.quantile(values, q, method=method))
    except TypeError:
        # NumPy<1.22 compatibility.
        return float(np.quantile(values, q, interpolation=method))


def _best_pareto_balance_model(
    pareto: pd.DataFrame,
    *,
    metric_col: str = "rmse",
) -> pd.Series | None:
    if pareto.empty:
        return None
    work = pareto.copy()
    work["log_latency"] = np.log10(np.maximum(work["mean_inference_us"].to_numpy(dtype=float), 1e-12))
    work["balance_score"] = _normalized_distance_to_ideal(work, "log_latency", metric_col)
    best = work.sort_values(
        ["balance_score", metric_col, "mean_inference_us"],
        ascending=[True, True, True],
    ).iloc[0]
    return best


def _pareto_summary_lines(
    frame: pd.DataFrame,
    pareto: pd.DataFrame,
    *,
    metric_col: str = "rmse",
    metric_label: str = "RMSE",
) -> list[str]:
    lines: list[str] = []
    if frame.empty:
        return lines

    fastest = frame.loc[frame["mean_inference_us"].idxmin()]
    best_metric = frame.loc[frame[metric_col].idxmin()]
    best_balance = _best_pareto_balance_model(pareto, metric_col=metric_col)

    lines.append(
        f"Fastest model: {fastest['model']} "
        f"({float(fastest['mean_inference_us']):.3f} us, {metric_label} {float(fastest[metric_col]):.4f})."
    )
    lines.append(
        f"Lowest-{metric_label} model: {best_metric['model']} "
        f"({metric_label} {float(best_metric[metric_col]):.4f}, {float(best_metric['mean_inference_us']):.3f} us)."
    )

    if best_balance is not None:
        lines.append(
            f"Best Pareto-balance model: {best_balance['model']} "
            f"({metric_label} {float(best_balance[metric_col]):.4f}, {float(best_balance['mean_inference_us']):.3f} us)."
        )

    lines.append(f"Pareto-optimal models: {len(pareto)} of {len(frame)}.")
    return lines[:4]


def _adaptive_metric_limits(frame: pd.DataFrame, value_col: str) -> tuple[float, float, float, pd.Series]:
    raw_values = frame[value_col].to_numpy(dtype=float)
    finite_mask = np.isfinite(raw_values)
    finite_values = raw_values[finite_mask]
    if finite_values.size == 0:
        return 0.0, 1.0, 1.0, pd.Series(False, index=frame.index)

    values_sorted = np.sort(finite_values)
    min_val = float(np.min(values_sorted))
    max_val = float(np.max(values_sorted))
    n = len(values_sorted)

    if n < 4:
        span = max(max_val - min_val, abs(max_val) * 0.08, 0.01)
        pad = span * 0.15
        lower = max(0.0, min_val - pad)
        upper = max_val + pad
        outlier_mask = pd.Series(np.zeros(len(frame), dtype=bool), index=frame.index)
        return lower, upper, max_val, outlier_mask

    q1 = _safe_quantile(values_sorted, 0.25, method="lower")
    q3 = _safe_quantile(values_sorted, 0.75, method="lower")
    iqr = float(q3 - q1)
    p90 = _safe_quantile(values_sorted, 0.90, method="lower")
    p95 = _safe_quantile(values_sorted, 0.95, method="lower")

    median = float(np.median(values_sorted))
    mad = float(np.median(np.abs(values_sorted - median)))
    robust_sigma = 1.4826 * max(mad, 1e-12)
    mad_upper = median + 4.0 * robust_sigma

    second_max = float(values_sorted[-2])
    spread = max(second_max - min_val, 1e-12)
    gap_to_top = max_val - second_max

    robust_upper = max(second_max, q3 + 1.5 * max(iqr, 1e-12), mad_upper, p90)
    robust_upper = min(max_val, robust_upper)
    if not np.isfinite(robust_upper):
        robust_upper = max_val

    clip_needed = (
        max_val > robust_upper
        and (
            gap_to_top > 0.35 * spread
            or max_val > p95 * 1.05
            or max_val > median + 6.0 * robust_sigma
        )
    )
    clip_cap = robust_upper if clip_needed else max_val

    visible = np.minimum(values_sorted, clip_cap)
    vis_min = float(np.min(visible))
    vis_max = float(np.max(visible))
    span = max(vis_max - vis_min, abs(vis_max) * 0.08, 0.01)
    lower = max(0.0, vis_min - span * 0.15)
    upper = vis_max + span * 0.18

    outlier_mask = pd.Series(np.isfinite(raw_values) & (raw_values > clip_cap), index=frame.index)
    return lower, upper, clip_cap, outlier_mask


def _adaptive_metric_clip(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr, np.zeros_like(arr, dtype=bool), 1.0

    sorted_vals = np.sort(finite)
    max_val = float(sorted_vals[-1])
    n = len(sorted_vals)
    if n < 4:
        return np.minimum(arr, max_val), arr > max_val, max_val

    q1 = _safe_quantile(sorted_vals, 0.25, method="lower")
    q3 = _safe_quantile(sorted_vals, 0.75, method="lower")
    iqr = float(q3 - q1)
    p90 = _safe_quantile(sorted_vals, 0.90, method="lower")
    p95 = _safe_quantile(sorted_vals, 0.95, method="lower")
    median = float(np.median(sorted_vals))
    mad = float(np.median(np.abs(sorted_vals - median)))
    robust_sigma = 1.4826 * max(mad, 1e-12)

    second_max = float(sorted_vals[-2])
    robust_upper = max(second_max, q3 + 1.5 * max(iqr, 1e-12), p90, median + 4.5 * robust_sigma)
    robust_upper = min(robust_upper, max_val)

    clip_needed = (
        max_val > robust_upper
        and (max_val > p95 * 1.04 or max_val > median + 6.0 * robust_sigma)
    )
    clip_cap = robust_upper if clip_needed else max_val
    plot_vals = np.minimum(arr, clip_cap)
    outlier_mask = np.isfinite(arr) & (arr > clip_cap)
    return plot_vals, outlier_mask, clip_cap

def figure_pareto_frontier(
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame | None,
    output_dir: Path,
    dpi: int,
) -> bool:
    summary = model_summary_frame(metrics, predictions)
    if summary.empty:
        warn("Skipping pareto_frontier.png: no model summary is available.")
        return False

    frame = summary.dropna(subset=["model", "mean_inference_us", "qlike"]).copy()
    frame = frame[frame["mean_inference_us"] > 0].copy()
    if frame.empty:
        warn("Skipping pareto_frontier.png: no model has positive latency and QLIKE values.")
        return False

    optimal_mask = pareto_optimal_mask(frame, x_col="mean_inference_us", y_col="qlike")
    pareto = frame[optimal_mask].sort_values("mean_inference_us")
    dominated = frame[~optimal_mask].sort_values("mean_inference_us")
    y_lower, y_upper, clip_cap, outlier_mask = _adaptive_metric_limits(frame, "qlike")
    frame = frame.copy()
    frame["qlike_plot"] = np.minimum(frame["qlike"], clip_cap)
    pareto = frame[optimal_mask].sort_values("mean_inference_us")
    dominated = frame[~optimal_mask].sort_values("mean_inference_us")

    fig, ax = styled_subplots(figsize=(10.0, 6.0), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.90)
    style_axis(ax, which="both", alpha=0.40)

    dominated_scatter = None
    if not dominated.empty:
        dominated_scatter = ax.scatter(
            dominated["mean_inference_us"],
            dominated["qlike_plot"],
            color="gray",
            s=50,
            alpha=0.7,
            label="Dominated models",
            zorder=2,
        )

    frontier_line = None
    pareto_scatter = None
    if not pareto.empty:
        (frontier_line,) = ax.plot(
            pareto["mean_inference_us"],
            pareto["qlike_plot"],
            color="red",
            linewidth=1.5,
            label="Pareto frontier",
            zorder=3,
        )
        pareto_scatter = ax.scatter(
            pareto["mean_inference_us"],
            pareto["qlike_plot"],
            color="red",
            s=60,
            label="Pareto optimal models",
            zorder=4,
        )

    label_df = frame.sort_values(["mean_inference_us", "qlike"], ascending=[True, True]).reset_index(drop=False)
    offsets = _label_offsets(len(label_df))
    for idx, row in label_df.iterrows():
        original_idx = int(row["index"])
        color = "red" if bool(optimal_mask.loc[original_idx]) else "gray"
        outlier_suffix = f" ({float(row['qlike']):.3f})" if bool(outlier_mask.loc[original_idx]) else ""
        ax.annotate(
            f"{row['model']}{outlier_suffix}",
            (float(row["mean_inference_us"]), float(row["qlike_plot"])),
            xytext=offsets[idx],
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

    for x, color, text in [
        (1.0, COLOR_THEME["latency_1us"], "1 μs (HFT)"),
        (10.0, COLOR_THEME["latency_10us"], "10 μs (desk trading)"),
        (1000.0, COLOR_THEME["latency_1000us"], "1000 μs"),
    ]:
        ax.axvline(x=x, color=color, linestyle="--", linewidth=1.2, label="_nolegend_")
    for x, color in [(1.0, "green"), (10.0, "gold"), (1000.0, "purple")]:
        ax.axvline(x=x, color=color, linestyle="--", linewidth=1.2, label="_nolegend_")

    ax.set_xscale("log")
    ax.set_xlabel("Inference Time (μs)")
    ax.set_xlabel("Inference Time (\u03bcs)")
    ax.set_ylabel("QLIKE (lower is better)")
    ax.set_title("Pareto Frontier: Volatility Model Selection", pad=12)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    x_min = float(frame["mean_inference_us"].min())
    x_max = float(frame["mean_inference_us"].max())
    x_lower = max(min(x_min * 0.6, 1.0), 1e-6)
    x_upper = max(x_max * 2.0, 1000.0)
    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    clipped_handle = None
    if outlier_mask.any():
        outliers = frame[outlier_mask]
        if not outliers.empty:
            outlier_colors = [
                "red" if bool(optimal_mask.loc[idx]) else "gray"
                for idx in outliers.index
            ]
            clipped_handle = ax.scatter(
                outliers["mean_inference_us"],
                np.full(len(outliers), clip_cap),
                marker="^",
                s=90,
                c=outlier_colors,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
                label="Clipped for display",
            )

    from matplotlib.lines import Line2D

    latency_handles = [
        Line2D([0], [0], color="green", linestyle="--", linewidth=1.2, label="1 μs (HFT)"),
        Line2D([0], [0], color="gold", linestyle="--", linewidth=1.2, label="10 μs (desk trading)"),
        Line2D([0], [0], color="purple", linestyle="--", linewidth=1.2, label="1000 μs"),
    ]
    latency_handles = [
        Line2D([0], [0], color="green", linestyle="--", linewidth=1.2, label="1 \u03bcs (HFT)"),
        Line2D([0], [0], color="gold", linestyle="--", linewidth=1.2, label="10 \u03bcs (desk trading)"),
        Line2D([0], [0], color="purple", linestyle="--", linewidth=1.2, label="1000 \u03bcs"),
    ]
    legend_handles = [h for h in [frontier_line, pareto_scatter, dominated_scatter] if h is not None]
    legend_handles.extend(latency_handles)
    if clipped_handle is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="^",
                linestyle="None",
                markerfacecolor="gray",
                markeredgecolor="black",
                markersize=8,
                label="Clipped for display",
            )
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    save_figure(fig, output_dir / "pareto_frontier.png", dpi)
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
    frame["model"] = frame["model"].astype(str)
    frame = frame.sort_values("rmse", na_position="last")

    has_qlike = "qlike" in frame.columns and frame["qlike"].notna().any()
    if has_qlike:
        fig, axes = styled_subplots(
            nrows=1,
            ncols=2,
            figsize=(14.6, 6.2),
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.12, wspace=0.23)
        ax_rmse, ax_qlike = axes
        rmse_frame = frame.dropna(subset=["rmse"]).sort_values("rmse", ascending=True)
        qlike_frame = frame.dropna(subset=["qlike"]).sort_values("qlike", ascending=True)

        y_rmse = np.arange(len(rmse_frame), dtype=float)
        rmse_actual = rmse_frame["rmse"].to_numpy(dtype=float)
        rmse_plot, rmse_outlier_mask, _ = _adaptive_metric_clip(rmse_actual)
        ax_rmse.hlines(
            y=y_rmse,
            xmin=0.0,
            xmax=rmse_plot,
            color="#9fb3c8",
            linewidth=2.1,
            alpha=0.8,
        )
        ax_rmse.scatter(
            rmse_plot,
            y_rmse,
            s=170,
            color=COLOR_THEME["primary"],
            edgecolors="#102a43",
            linewidths=1.1,
            zorder=3,
        )
        if rmse_outlier_mask.any():
            ax_rmse.scatter(
                rmse_plot[rmse_outlier_mask],
                y_rmse[rmse_outlier_mask],
                marker="^",
                s=185,
                color=COLOR_THEME["pareto"],
                edgecolors="#102a43",
                linewidths=1.0,
                zorder=4,
            )
        ax_rmse.set_yticks(y_rmse)
        ax_rmse.set_yticklabels(rmse_frame["model"])
        ax_rmse.invert_yaxis()
        ax_rmse.set_title("RMSE by Model")
        ax_rmse.set_xlabel("RMSE", fontsize=13)
        ax_rmse.tick_params(axis="both", labelsize=11)
        style_axis(ax_rmse, axis="x")
        if rmse_plot.size:
            rmse_upper = max(float(np.max(rmse_plot)) * 1.18, 1.0)
            ax_rmse.set_xlim(0.0, rmse_upper)
            label_pad = max(rmse_upper * 0.018, 0.04)
            for y, x_plot, x_true, is_outlier in zip(y_rmse, rmse_plot, rmse_actual, rmse_outlier_mask):
                text = f"{float(x_true):.4f}" + ("*" if bool(is_outlier) else "")
                x_text = float(x_plot + label_pad)
                ha = "left"
                if x_text > rmse_upper * 0.985:
                    x_text = float(x_plot - label_pad * 1.1)
                    ha = "right"
                ax_rmse.text(
                    x_text,
                    float(y),
                    text,
                    va="center",
                    ha=ha,
                    fontsize=11,
                    color=COLOR_THEME["ink"],
                    bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.7},
                )
            if rmse_outlier_mask.any():
                ax_rmse.text(
                    0.01,
                    0.03,
                    "* value clipped for display",
                    transform=ax_rmse.transAxes,
                    fontsize=9,
                    color=COLOR_THEME["muted"],
                )

        y_qlike = np.arange(len(qlike_frame), dtype=float)
        qlike_actual = qlike_frame["qlike"].to_numpy(dtype=float)
        qlike_plot, qlike_outlier_mask, _ = _adaptive_metric_clip(qlike_actual)
        ax_qlike.hlines(
            y=y_qlike,
            xmin=0.0,
            xmax=qlike_plot,
            color="#9fb3c8",
            linewidth=2.1,
            alpha=0.8,
        )
        ax_qlike.scatter(
            qlike_plot,
            y_qlike,
            s=170,
            color=COLOR_THEME["secondary"],
            edgecolors="#102a43",
            linewidths=1.1,
            zorder=3,
        )
        if qlike_outlier_mask.any():
            ax_qlike.scatter(
                qlike_plot[qlike_outlier_mask],
                y_qlike[qlike_outlier_mask],
                marker="^",
                s=185,
                color=COLOR_THEME["pareto"],
                edgecolors="#102a43",
                linewidths=1.0,
                zorder=4,
            )
        ax_qlike.set_yticks(y_qlike)
        ax_qlike.set_yticklabels(qlike_frame["model"])
        ax_qlike.invert_yaxis()
        ax_qlike.set_title("QLIKE by Model")
        ax_qlike.set_xlabel("QLIKE", fontsize=13)
        ax_qlike.tick_params(axis="both", labelsize=11)
        style_axis(ax_qlike, axis="x")
        if qlike_plot.size:
            qlike_upper = max(float(np.max(qlike_plot)) * 1.18, 1.0)
            ax_qlike.set_xlim(0.0, qlike_upper)
            label_pad = max(qlike_upper * 0.018, 0.025)
            for y, x_plot, x_true, is_outlier in zip(y_qlike, qlike_plot, qlike_actual, qlike_outlier_mask):
                text = f"{float(x_true):.4f}" + ("*" if bool(is_outlier) else "")
                x_text = float(x_plot + label_pad)
                ha = "left"
                if x_text > qlike_upper * 0.985:
                    x_text = float(x_plot - label_pad * 1.1)
                    ha = "right"
                ax_qlike.text(
                    x_text,
                    float(y),
                    text,
                    va="center",
                    ha=ha,
                    fontsize=11,
                    color=COLOR_THEME["ink"],
                    bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.7},
                )
            if qlike_outlier_mask.any():
                ax_qlike.text(
                    0.01,
                    0.03,
                    "* value clipped for display",
                    transform=ax_qlike.transAxes,
                    fontsize=9,
                    color=COLOR_THEME["muted"],
                )

        ax_rmse.set_ylabel("Model", fontsize=13)
        ax_qlike.set_ylabel("Model", fontsize=13)
        fig.suptitle("Model Metric Comparison", y=0.96, fontsize=19, color=COLOR_THEME["ink"])
    else:
        rmse_frame = frame.dropna(subset=["rmse"]).sort_values("rmse", ascending=True)
        fig, ax = styled_subplots(figsize=(10.2, 6.0), constrained_layout=False)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.14)
        y = np.arange(len(rmse_frame), dtype=float)
        ax.hlines(
            y=y,
            xmin=0.0,
            xmax=rmse_frame["rmse"].to_numpy(dtype=float),
            color="#9fb3c8",
            linewidth=2.2,
            alpha=0.8,
        )
        ax.scatter(
            rmse_frame["rmse"],
            y,
            s=180,
            color=COLOR_THEME["primary"],
            edgecolors="#102a43",
            linewidths=1.1,
            zorder=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(rmse_frame["model"])
        ax.invert_yaxis()
        ax.set_title("Model RMSE Comparison")
        ax.set_xlabel("RMSE", fontsize=13)
        ax.set_ylabel("Model", fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        style_axis(ax, axis="x")
        annotate_lollipop_values(
            ax,
            y,
            rmse_frame["rmse"].to_numpy(dtype=float),
            fmt=".4f",
            pad_fraction=0.018,
            fontsize=11,
        )
        if not rmse_frame.empty:
            rmse_max = float(rmse_frame["rmse"].max())
            ax.set_xlim(0.0, rmse_max * 1.2 if rmse_max > 0 else 1.0)

    save_figure(fig, output_dir / "model_metric_comparison.png", dpi)
    return True


def choose_representative_stock_for_series(predictions: pd.DataFrame) -> str | None:
    required = {"stock_id", "time_id", "model"}
    if not required.issubset(predictions.columns):
        return None
    group = (
        predictions.groupby("stock_id", as_index=False)
        .agg(
            n_times=("time_id", "nunique"),
            n_models=("model", "nunique"),
            n_rows=("time_id", "size"),
        )
        .sort_values(["n_times", "n_models", "n_rows", "stock_id"], ascending=[False, False, False, True])
    )
    if group.empty:
        return None
    row = group.iloc[0]
    return str(row["stock_id"])


def _model_rmse_for_single_stock(stock_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        stock_frame.groupby("model", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    g["pred_vol"].to_numpy(dtype=float)
                                    - g["actual_vol"].to_numpy(dtype=float)
                                )
                                ** 2
                            )
                        )
                    )
                }
            )
        )
        .reset_index(drop=True)
        .assign(model=lambda d: d["model"].astype(str))
        .sort_values(["rmse", "model"], ascending=[True, True])
    )


def _fastest_remaining_model(raw_frame: pd.DataFrame, excluded: set[str]) -> str | None:
    if "inference_ms" not in raw_frame.columns:
        return None
    speed = raw_frame[["model", "inference_ms"]].copy()
    speed["model"] = speed["model"].astype(str)
    speed["inference_ms"] = pd.to_numeric(speed["inference_ms"], errors="coerce")
    speed = speed.replace([np.inf, -np.inf], np.nan).dropna(subset=["inference_ms"])
    if speed.empty:
        return None
    ranked = (
        speed.groupby("model", as_index=False)["inference_ms"]
        .mean()
        .sort_values(["inference_ms", "model"], ascending=[True, True])
    )
    for model_name in ranked["model"].astype(str):
        if model_name not in excluded:
            return model_name
    return None

#select representative models for plotting when too many models exist
def _select_models_for_single_stock(
    stock_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    *,
    min_models: int = 3,
    max_models: int = 5,
    preferred_top_k: int = 4,
) -> list[str]:
    metric = _model_rmse_for_single_stock(stock_frame)
    if metric.empty:
        return []

    available_models = metric["model"].astype(str).tolist()
    n_models = len(available_models)
    if n_models <= max_models:
        return available_models

    top_k = max(min_models, min(max_models - 1, preferred_top_k))
    top_k = min(top_k, n_models)
    selected = available_models[:top_k]

    baseline_candidate = _fastest_remaining_model(raw_frame, excluded=set(selected))
    if baseline_candidate and baseline_candidate not in selected:
        selected.append(baseline_candidate)

    return selected[:max_models]


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

    selected_stock_id = choose_representative_stock_for_series(predictions)
    if selected_stock_id is None:
        warn("Skipping single_stock_actual_vs_predicted.png: could not pick a representative stock.")
        return False

    subset = predictions[
        predictions["stock_id"].astype(str) == selected_stock_id
    ].copy()
    subset["time_id"] = pd.to_numeric(subset["time_id"], errors="coerce")
    subset["actual_vol"] = pd.to_numeric(subset["actual_vol"], errors="coerce")
    subset["pred_vol"] = pd.to_numeric(subset["pred_vol"], errors="coerce")
    subset = subset.dropna(subset=["time_id", "actual_vol", "pred_vol"])
    if subset.empty:
        warn("Skipping single_stock_actual_vs_predicted.png: selected stock has no finite values.")
        return False

    grouped = (
        subset.groupby(["time_id", "model"], as_index=False)
        .agg(actual_vol=("actual_vol", "mean"), pred_vol=("pred_vol", "mean"))
        .sort_values(["time_id", "model"])
    )
    if grouped.empty:
        warn("Skipping single_stock_actual_vs_predicted.png: aggregated time-series data is empty.")
        return False

    actual_series = (
        grouped.groupby("time_id", as_index=False)
        .agg(actual_vol=("actual_vol", "mean"))
        .sort_values("time_id")
    )
    if actual_series["time_id"].nunique() < 2:
        warn("Skipping single_stock_actual_vs_predicted.png: not enough time points for a time-series chart.")
        return False

    top_models = _select_models_for_single_stock(
        grouped,
        subset,
        min_models=3,
        max_models=5,
        preferred_top_k=4,
    )
    if not top_models:
        warn("Skipping single_stock_actual_vs_predicted.png: no models available for plotting.")
        return False

    model_palette = model_color_map(top_models)
    fig, ax = styled_subplots(figsize=(12.2, 6.8), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.79, top=0.88, bottom=0.13)
    style_axis(ax, axis="y", alpha=0.22)
    ax.grid(True, axis="x", alpha=0.10, linewidth=0.7)

    actual_plot = downsample_for_plot(actual_series, max_points=900).copy()
    actual_plot["actual_vol_smoothed"] = gentle_smooth(
        actual_plot["actual_vol"],
        target_points=len(actual_plot),
        scale=55,
        min_window=5,
        max_window=33,
    )

    ax.plot(
        actual_plot["time_id"],
        actual_plot["actual_vol_smoothed"],
        label="Actual realised volatility",
        color=COLOR_THEME["actual_line"],
        linewidth=1.3,
        alpha=0.96,
        zorder=4,
    )

    for idx, model in enumerate(top_models):
        model_slice = grouped[grouped["model"].astype(str) == str(model)].sort_values("time_id")
        if model_slice.empty:
            continue
        model_plot = downsample_for_plot(model_slice, max_points=900).copy()
        model_plot["pred_vol_smoothed"] = gentle_smooth(
            model_plot["pred_vol"],
            target_points=len(model_plot),
            scale=55,
            min_window=5,
            max_window=33,
        )
        linestyle = MODEL_LINESTYLES[idx % len(MODEL_LINESTYLES)]
        ax.plot(
            model_plot["time_id"],
            model_plot["pred_vol_smoothed"],
            label=f"Predicted ({model})",
            color=model_palette[str(model)],
            linewidth=1.3,
            alpha=0.78,
            linestyle=linestyle,
            zorder=2,
        )

    ax.set_ylabel("Volatility", fontsize=12)
    ax.set_xlabel("Trading Window Index", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_title(f"Single-Stock Volatility Forecast Comparison (stock_id={selected_stock_id})", pad=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=9, integer=True))
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        fontsize=10.5,
        framealpha=0.96,
    )
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

    fig, ax = styled_subplots(figsize=(9.4, 5.3))
    bars = ax.bar(counts.index, counts.values, color=COLOR_THEME["secondary"], edgecolor="white", linewidth=0.8)
    ax.set_title("Best Model Counts Across Stocks")
    ax.set_ylabel("Count of stocks")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=25)
    style_axis(ax, axis="y", alpha=0.32)
    max_count = float(counts.max())
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(value) + max(max_count * 0.012, 0.5),
            str(int(value)),
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR_THEME["ink"],
        )
    ax.set_ylim(0.0, max_count * 1.12 if max_count > 0 else 1.0)
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

    fig, ax = styled_subplots(figsize=(10.2, 6.0))
    bars = ax.barh(top10["feature"], top10["importance"], color=COLOR_THEME["feature"], alpha=0.92)
    ax.set_title("Top 10 Average Feature Importances")
    ax.set_xlabel("Average importance")
    ax.set_ylabel("Feature")
    style_axis(ax, axis="x", alpha=0.32)
    max_imp = float(top10["importance"].max())
    for bar in bars:
        x = float(bar.get_width())
        y = float(bar.get_y() + bar.get_height() / 2.0)
        ax.text(
            x + max(max_imp * 0.015, 1e-4),
            y,
            f"{x:.3f}",
            va="center",
            ha="left",
            fontsize=8.8,
            color=COLOR_THEME["ink"],
        )
    ax.set_xlim(0.0, max_imp * 1.15 if max_imp > 0 else 1.0)
    save_figure(fig, output_dir / "feature_importance_top10.png", dpi)
    return True


def main() -> int:
    args = parse_args()
    apply_global_style()
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
    
    #list of report figures generated from run artifact
    figure_tasks: list[tuple[str, Callable[[], bool]]] = [
        (
            "pareto_frontier.png",
            lambda: figure_pareto_frontier(metrics, predictions, output_dir, args.dpi),
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